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
from . import numerics as mhd_num
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
            if bc not in allowed_fluid:
                raise ValueError(
                    f"boundary.fluid[{i}] = {bc!r} must be one of "
                    f"{sorted(allowed_fluid)}")
        allowed_field = set(_core.mhd.registered_mhd_field_boundaries())
        for i, bc in enumerate(self.boundary.field):
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
        analytic_mode = bg.file is None and bg.a_file is None
        # A profile is consumed only in analytic mode. Explicit file modes
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
            if bg.a_file is None:
                raise ValueError(
                    "background_field.params.vacuum_project requires "
                    "background_field.a_file")
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
        if bg.file is not None and bg.a_file is not None:
            raise ValueError(
                "background_field: set at most one of 'file' (staggered B0 arrays) "
                "or 'a_file' (coil vector-potential A); they are mutually exclusive")
        if bg.file is not None and bg.params:
            raise ValueError(
                "background_field.params is not used with background_field.file")
        if bg.a_file is not None:
            unknown = set(bg.params) - {"b_scale", "vacuum_project"}
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(
                    "unknown background_field.params key(s) for a_file: " + names)
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
        # Radial finite-volume averages use the ring measure r dr, whereas the
        # current MP5/MP7 reconstruction and magnetic collocation coefficients
        # are uniform Cartesian moments. Accepting either MP scheme here would
        # advertise its Cartesian design order while retaining only second-order
        # radial consistency. Keep the supported scheme explicit until native
        # reconstruction owns radius-dependent weighted moments.
        if self.numerics.reconstruction != "muscl_minmod":
            raise ValueError(
                "geometry 'cylindrical' currently supports only "
                "numerics.reconstruction='muscl_minmod'; MP5/MP7 require "
                "r-weighted radial finite-volume moments")
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
            "a_file"}, "background_field")
    enabled = _as_bool(d.get("enabled", False), "background_field.enabled")
    # `enabled` is a true master switch: ignored values in a disabled block do
    # not participate in parsing, registry lookup, unit conversion, or native
    # validation. Canonicalize it to the same defaults as an absent block.
    if not enabled:
        return BackgroundConfig()
    file_raw = d.get("file")
    a_file_raw = d.get("a_file")
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
# Each generator returns a dict of full ghost-padded host buffers (one per
# STATE_COMPONENT), in the solver's storage layout (pitch = nx + 2*nghost,
# height = ny + 2*nghost, row-major reshape(-1)), ready for solver.seed_state().
#
# Storage convention (mirrors C++ MhdField2D / the solver seed contract):
#   * rho, mx, my, mz, energy : cell-centered conserved quantities.
#   * bx, by                  : seeded into the FACE-staggered slots (bx_face,
#                               by_face); for a UNIFORM or x-only-varying field
#                               the cell-centered analytic value sampled at the
#                               left/bottom face equals the cell value to the
#                               accuracy these smooth ICs need, and div B stays
#                               ~0. Where the analytic B varies, the face value
#                               is sampled at the face location (documented per
#                               token below).
#   * bz                      : cell-centered (bz_cell), out-of-plane toroidal.
#
# The energy uses mhd_num.primitive_to_energy (the single p->E source of truth).


def _padded_grids(domain: Domain, nghost: int):
    """Return (cell-center x, cell-center y) meshes over the FULL padded storage.

    Index (j_pad, i_pad) maps to interior cell (i, j) = (i_pad - g, j_pad - g);
    cell-center coordinate uses the same origin + (idx + 0.5)*d convention as the
    C++ Grid2D, evaluated at the (possibly negative) interior index so ghost
    cells carry a consistent analytic value.
    """
    nx, ny = domain.nx, domain.ny
    g = _as_exact_int(nghost, "nghost")
    if g < 0:
        raise ValueError("nghost must be non-negative")
    pitch = nx + 2 * g
    height = ny + 2 * g
    dx = domain.lx_m / nx
    dy = domain.ly_m / ny
    i_pad = np.arange(pitch)
    j_pad = np.arange(height)
    ii, jj = np.meshgrid(i_pad, j_pad)            # (height, pitch)
    i_int = ii - g
    j_int = jj - g
    xc = domain.origin_x_m + (i_int + 0.5) * dx   # cell-center x
    yc = domain.origin_y_m + (j_int + 0.5) * dy   # cell-center y
    xf = domain.origin_x_m + i_int * dx           # left-face x (for bx_face)
    yf = domain.origin_y_m + j_int * dy           # bottom-face y (for by_face)
    if not all(np.all(np.isfinite(a)) for a in (xc, yc, xf, yf)):
        raise ValueError("padded grid coordinates are not representable in float64")
    return xc, yc, xf, yf, dx, dy


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


def _empty_state(shape) -> dict:
    return {name: np.zeros(shape, dtype=np.float64) for name in STATE_COMPONENTS}


def _pack(state: dict) -> dict:
    """Flatten each (height, pitch) buffer to the 1-D layout seed_state wants."""
    return {name: arr.reshape(-1) for name, arr in state.items()}


def build_initial_state(deck: MhdDeck, nghost: int) -> dict:
    """Build the seeded conserved state for ``deck.initial.type``.

    Returns ``{component: 1-D host buffer}`` for every STATE_COMPONENT, ready to
    hand to ``solver.seed_state(component, buf)``.
    """
    _validate_padded_cylindrical_domain(deck, nghost)
    builders = {
        "brio_wu": _ic_brio_wu,
        "alfven_wave": _ic_alfven_wave,
        "orszag_tang": _ic_orszag_tang,
        "blast": _ic_blast,
        "rotor": _ic_rotor,
        "confined_blob": _ic_confined_blob,
    }
    builder = builders[deck.initial.type]
    state = builder(deck, nghost)

    # Builders express their analytic magnetic profiles in deck units and, for
    # historical reasons, initially form energy from the raw face slots.  Convert
    # SI Tesla to the solver's mu0=1 magnetic variable and replace that magnetic
    # energy by the field collocated at the cell centre.
    shape = (deck.domain.ny + 2 * nghost, deck.domain.nx + 2 * nghost)
    bx_deck = np.asarray(state["bx"], dtype=np.float64).reshape(shape)
    by_deck = np.asarray(state["by"], dtype=np.float64).reshape(shape)
    bz_deck = np.asarray(state["bz"], dtype=np.float64).reshape(shape)
    energy = np.asarray(state["energy"], dtype=np.float64).reshape(shape)
    old_magnetic = mhd_num.half_squared_norm3(bx_deck, by_deck, bz_deck)

    bx = mhd_units.magnetic_to_internal(bx_deck, deck.units)
    by = mhd_units.magnetic_to_internal(by_deck, deck.units)
    bz = mhd_units.magnetic_to_internal(bz_deck, deck.units)
    # Face data are finite-volume face averages, not point values. Match the
    # native order-aware face-to-cell quadrature (including its outer-ghost
    # closure) so the seeded energy and the solver EOS use exactly the same B.
    bx_c = mhd_num.face_samples_to_cell_average(bx, axis=1, nghost=nghost)
    by_c = mhd_num.face_samples_to_cell_average(by, axis=0, nghost=nghost)
    new_magnetic = mhd_num.half_squared_norm3(bx_c, by_c, bz)

    state["bx"] = bx.reshape(-1)
    state["by"] = by.reshape(-1)
    state["bz"] = bz.reshape(-1)
    adjusted_energy = energy - old_magnetic + new_magnetic
    state["energy"] = adjusted_energy.reshape(-1)

    # Final preflight on the exact conserved arrays handed to the native
    # solver.  Face-to-cell magnetic collocation and float64 energy assembly can
    # expose an otherwise-hidden loss of internal energy at extreme scales; do
    # not defer that error to the first CFL reduction/device kernel.
    rho = np.asarray(state["rho"], dtype=np.float64).reshape(shape)
    mx = np.asarray(state["mx"], dtype=np.float64).reshape(shape)
    my = np.asarray(state["my"], dtype=np.float64).reshape(shape)
    mz = np.asarray(state["mz"], dtype=np.float64).reshape(shape)
    if not np.all(np.isfinite(rho)) or np.any(rho <= 0.0):
        raise ValueError(
            "initial state must have finite, strictly positive density everywhere")
    pressure = mhd_num.conserved_to_pressure(
        rho, mx, my, mz, adjusted_energy, bx_c, by_c, bz, deck.numerics.gamma)
    if np.any(pressure <= 0.0):
        raise ValueError(
            "initial state must have strictly positive gas pressure everywhere")
    return state


# =============================================================================
# Static background magnetic field B0 (field-split form B = B0 + b)
# =============================================================================

# The optional cylindrical vacuum projection solves a curl-free elliptic
# condition.  Its algebraic stopping tolerance is independent of the strict,
# scale-free solenoidality proof applied to the resulting staggered field.
_VACUUM_PROJECTION_RELATIVE_TOL = 5.0e-11


def build_background_field(deck: MhdDeck, nghost: int) -> Union[dict, None]:
    """Build the static background field B0 for ``deck.background``.

    Returns ``{"b0x": buf, "b0y": buf, "b0z": buf}`` of 1-D ghost-padded host
    buffers (length ``(nx+2g)*(ny+2g)``, row-major like the IC builders), ready
    for ``solver.seed_background(component, buf)``; or ``None`` when the
    background is disabled (the solver then runs its zero-B0 fast path).

    Three construction modes (mutually selected by the deck):

    * **uniform** (``profile == "uniform"``, no ``file``): constant components
      ``b0x == bx0``, ``b0y == by0``, ``b0z == bz0`` everywhere.
    * **profile** (named ``profile``, no ``file``): sample the analytic profile
      over the padded staggered meshes from :func:`_padded_grids` (b0x at the
      left face ``xf``, b0y at the bottom face ``yf``, b0z at cell centers).
      Registered profile parameters are forwarded to the native sampler.
    * **file** (``file:`` given): ``np.load`` the npz and read arrays ``b0x``,
      ``b0y``, ``b0z`` each shaped ``(ny+2g, nx+2g)`` or flat ``(storage,)``;
      reshape/flatten to the 1-D storage layout.

    In ALL modes the interior discrete face-divergence of the assembled field is
    checked and a non-divergence-free background raises ``ValueError``.
    """
    _validate_padded_cylindrical_domain(deck, nghost)
    bg = deck.background
    if not bg.enabled:
        return None

    nx, ny = deck.domain.nx, deck.domain.ny
    g = nghost
    pitch = nx + 2 * g
    height = ny + 2 * g
    storage = pitch * height
    shape = (height, pitch)
    dx = deck.domain.lx_m / nx
    dy = deck.domain.ly_m / ny

    if bg.a_file is not None:
        b0x, b0y, b0z = _background_from_a_file(deck, nghost, shape)
    elif bg.file is not None:
        b0x, b0y, b0z = _background_from_file(bg.file, deck.source_dir, shape,
                                              storage)
    else:
        b0x, b0y, b0z = _background_from_profile(deck, nghost, shape)

    # SI decks provide B0 in tesla (and A in T m); the solver stores B/sqrt(mu0)
    # so magnetic pressure and Alfven speed use the normalized mu0=1 equations.
    b0x = mhd_units.magnetic_to_internal(b0x, deck.units)
    b0y = mhd_units.magnetic_to_internal(b0y, deck.units)
    b0z = mhd_units.magnetic_to_internal(b0z, deck.units)

    # Use the same per-cell directional-derivative defect as the native live
    # and background preflights. Local Cartesian offsets cancel before the
    # common-exponent ratio, so a strong DC field cannot hide a represented
    # slope and there is no unit-dependent absolute floor.
    divb_defect = mhd_num.background_divergence_relative_linf(
        b0x, b0y, nx, ny, g, dx, dy,
        geometry=deck.geometry, origin_x=deck.domain.origin_x_m)
    if divb_defect > mhd_num.DISCRETE_SOLENOIDAL_TOLERANCE:
        raise ValueError(
            "background_field is not discretely divergence-free: maximum "
            f"relative stencil defect {divb_defect:.3e} exceeds "
            f"{mhd_num.DISCRETE_SOLENOIDAL_TOLERANCE:.3e}.")
    mhd_num.validate_background_boundary_compatibility(
        b0x, b0y, b0z, nx, ny, g, deck.boundary.field)

    return {"b0x": b0x.reshape(-1), "b0y": b0y.reshape(-1), "b0z": b0z.reshape(-1)}


def _project_cylindrical_vacuum_a(a_ji, r_face, dx, dy):
    """Project sampled ``A_phi`` onto the discrete annular vacuum operator.

    The projection fixes every value on the outer boundary of the padded corner
    grid and solves for ``psi = r A_phi`` at interior nodes so that

    ``D_r[(1/r) D_r psi] + D_zz(A_phi) = 0``.

    The radial differences are exactly the annular differences used to construct
    ``B_z`` below, and the axial differences are exactly those used for ``B_r``.
    Consequently the resulting face field has telescoping discrete ``div(B)``
    from the curl construction, while its discrete ``curl(B)`` is driven to the
    CG target by the solved equation. This is an opt-in operation because it
    replaces the interior by
    the unique discrete-vacuum harmonic continuation of the supplied boundary.
    Without projection the supplied potential is differenced directly; the
    resulting current-carrying field is valid if its divergence passes setup
    validation.

    A strictly positive padded radial interval is required. The r=0 parity
    closure needs a separate axis row in this elliptic operator and is therefore
    intentionally not inferred here.
    """
    a = np.asarray(a_ji, dtype=np.float64)
    radii = np.asarray(r_face, dtype=np.float64)
    if a.ndim != 2 or radii.ndim != 1 or a.shape[1] != radii.size:
        raise ValueError("invalid corner grid for cylindrical vacuum projection")
    if not np.all(radii > 0.0):
        raise ValueError(
            "background_field.params.vacuum_project requires the full padded "
            "corner grid to lie at r > 0 (annular geometry)")

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(radii)):
        raise ValueError("cylindrical vacuum projection inputs must be finite")
    if not math.isfinite(dx) or dx <= 0.0 or not math.isfinite(dy) or dy <= 0.0:
        raise ValueError("cylindrical vacuum projection spacing must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        psi = a * radii[None, :]
    if not np.all(np.isfinite(psi)):
        raise ValueError("r*A_phi is not representable in float64")
    r = radii[1:-1]
    dr_e = radii[2:] - r
    dr_w = r - radii[:-2]
    rmid_e = 0.5 * radii[2:] + 0.5 * r
    rmid_w = 0.5 * r + 0.5 * radii[:-2]
    ring_e = dr_e * rmid_e
    ring_w = dr_w * rmid_w
    if np.any(ring_e <= 0.0) or np.any(ring_w <= 0.0):
        raise ValueError(
            "cylindrical vacuum projection requires monotonically increasing "
            "positive radial faces")

    # -L is symmetric positive definite for fixed Dirichlet boundary values.
    ce = (1.0 / dx) / ring_e
    cw = (1.0 / dx) / ring_w
    cz = ((1.0 / dy) / dy) / r
    diag = ce + cw + 2.0 * cz
    if (not np.all(np.isfinite(ce)) or not np.all(np.isfinite(cw))
            or not np.all(np.isfinite(cz)) or not np.all(np.isfinite(diag))):
        raise ValueError(
            "cylindrical vacuum projection coefficients are not representable")

    def apply_zero_boundary(x):
        out = diag[None, :] * x
        out[:, :-1] -= ce[:-1][None, :] * x[:, 1:]
        out[:, 1:] -= cw[1:][None, :] * x[:, :-1]
        out[:-1, :] -= cz[None, :] * x[1:, :]
        out[1:, :] -= cz[None, :] * x[:-1, :]
        return out

    centre = psi[1:-1, 1:-1]
    a_raw = (diag[None, :] * centre
             - ce[None, :] * psi[1:-1, 2:]
             - cw[None, :] * psi[1:-1, :-2]
             - cz[None, :] * (psi[2:, 1:-1] + psi[:-2, 1:-1]))
    rhs = -a_raw

    # Scale the algebraic vacuum-operator residual to the characteristic field
    # derivative.  This is a curl/vacuum solve criterion, not the separate
    # scale-free divergence acceptance bound.
    br_raw = -(a[1:, :] - a[:-1, :]) / dy
    dr = radii[1:] - radii[:-1]
    rmid = 0.5 * radii[1:] + 0.5 * radii[:-1]
    bz_raw = ((a[:, 1:] - a[:, :-1]) / dr[None, :]
              + (0.5 * a[:, 1:] + 0.5 * a[:, :-1]) / rmid[None, :])
    field_scale = max(
        1.0,
        max(float(np.max(np.abs(br_raw))), float(np.max(np.abs(bz_raw)))) /
        min(dx, dy))
    if not math.isfinite(field_scale):
        raise ValueError("projected field derivative scale is not representable")
    target = _VACUUM_PROJECTION_RELATIVE_TOL * field_scale

    residual = rhs.copy()
    correction = np.zeros_like(residual)
    z = residual / diag[None, :]
    direction = z.copy()
    rz = float(np.sum(residual * z))
    converged = float(np.max(np.abs(residual))) <= target
    max_iterations = max(200, 20 * max(a.shape))
    for _ in range(max_iterations):
        if converged:
            break
        ad = apply_zero_boundary(direction)
        denom = float(np.sum(direction * ad))
        if not (math.isfinite(denom) and denom > 0.0 and math.isfinite(rz)):
            raise ValueError(
                "cylindrical vacuum projection failed: discrete operator lost "
                "positive definiteness")
        alpha = rz / denom
        correction += alpha * direction
        residual -= alpha * ad
        if float(np.max(np.abs(residual))) <= target:
            converged = True
            break
        z = residual / diag[None, :]
        rz_next = float(np.sum(residual * z))
        if not (math.isfinite(rz_next) and rz_next >= 0.0):
            raise ValueError(
                "cylindrical vacuum projection failed with a non-finite residual")
        beta = rz_next / rz
        direction = z + beta * direction
        rz = rz_next
    if not converged:
        final = float(np.max(np.abs(residual)))
        raise ValueError(
            "cylindrical vacuum projection did not converge within "
            f"{max_iterations} iterations (residual {final:.3e}, target "
            f"{target:.3e})")

    projected_psi = psi.copy()
    projected_psi[1:-1, 1:-1] += correction
    return projected_psi / radii[None, :]


def _background_from_a_file(deck: MhdDeck, nghost: int, shape):
    """Build a non-uniform, divergence-free B0 from a coil vector-potential npz.

    The coil CLI writes ``A_xyz_grid`` on the cell-corner grid of the FULL padded
    domain (so B0 is defined in the ghost layers too -- B0 is static and never
    ghost-refilled, and the reconstruction stencil reaches `nghost` cells past the
    interior). With the lab Y=0 slice and the mapping MHD-x = lab-X, MHD-y = lab-Z,
    out-of-plane = lab-Y, the saved array has shape ``(Ny+1, 1, Nx+1, 3)`` where
    ``Nx = nx + 2g``, ``Ny = ny + 2g`` are the padded cell counts. The in-plane B0
    is the discrete curl of the corner lab-Y component A[j, i]. In Cartesian
    geometry this is

        b0x_face(i,j) = -(A[j+1,i] - A[j,i]) / dy      # B_R on the left face
        b0y_face(i,j) =  (A[j,i+1] - A[j,i]) / dx      # B_z on the bottom face

    while cylindrical (R,Z) geometry uses the annular form

        b0y_face(i,j) = (R_hi A[j,i+1] - R_lo A[j,i])
                         / (0.5 (R_hi^2-R_lo^2)).

    spanning the full padded face layout, so the cell-centered discrete divergence
    telescopes to zero everywhere. The uniform out-of-plane ``bz0`` is added as the
    toroidal component. ``params.b_scale`` (default 1) scales the loaded A.
    """
    bg = deck.background
    nx, ny = deck.domain.nx, deck.domain.ny
    g = nghost
    height, pitch = shape                     # (ny+2g, nx+2g)
    dx = deck.domain.lx_m / nx
    dy = deck.domain.ly_m / ny

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
    b_scale = float(bg.params.get("b_scale", 1.0))
    if not math.isfinite(b_scale):
        raise ValueError("background_field.params.b_scale must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        a_ji = a_ji * b_scale
    if not np.all(np.isfinite(a_ji)):
        raise ValueError("scaled background vector potential is not representable")

    if bool(bg.params.get("vacuum_project", False)):
        # The opt-in projection is performed in deck units before B is converted
        # to the solver's B/sqrt(mu0) variable. The linear solve is scale
        # invariant; the derived field still undergoes the standard divergence
        # validation after unit conversion.
        face_index = np.arange(pitch + 1, dtype=np.float64) - g
        r_face = deck.domain.origin_x_m + face_index * dx
        a_ji = _project_cylindrical_vacuum_a(a_ji, r_face, dx, dy)

    b0x = -(a_ji[1:, :] - a_ji[:-1, :]) / dy  # (height, pitch+1) -> trim to faces
    if deck.geometry == "cylindrical":
        # Axisymmetric curl of A=A_phi e_phi:
        #   B_r = -dA_phi/dz,
        #   B_z = (1/r)d(r A_phi)/dr.
        # Use the same annular volume integral as the solver. This makes the
        # subsequent ring-weighted div(B0) telescope exactly.
        face_index = np.arange(pitch + 1, dtype=np.float64) - g
        r_face = deck.domain.origin_x_m + face_index * dx
        r_lo = r_face[:-1]
        r_hi = r_face[1:]
        dr = r_hi - r_lo
        r_mid = 0.5 * r_hi + 0.5 * r_lo
        ring = dr * r_mid
        if (not np.all(np.isfinite(ring)) or np.any(ring == 0.0)
                or np.any(dr <= 0.0)):
            raise ValueError(
                "background_field.a_file cylindrical curl encountered an invalid "
                "annular cell measure")
        b0y = ((a_ji[:, 1:] - a_ji[:, :-1]) / dr[None, :]
               + (0.5 * a_ji[:, 1:] + 0.5 * a_ji[:, :-1]) /
               r_mid[None, :])
    else:
        b0y = (a_ji[:, 1:] - a_ji[:, :-1]) / dx
    # b0x has shape (height,pitch+1); b0y has (height+1,pitch).
    # Face arrays are stored on the (height, pitch) cell layout: bx_face uses the
    # left face (drop the extra right column), by_face the bottom face (drop the
    # extra top row).
    b0x = b0x[:, :pitch].copy()
    b0y = b0y[:height, :].copy()
    b0z = np.full(shape, float(bg.bz0))
    if not all(np.all(np.isfinite(a)) for a in (b0x, b0y, b0z)):
        raise ValueError("background vector-potential curl is not representable")
    return b0x, b0y, b0z


def _background_from_profile(deck: MhdDeck, nghost: int, shape):
    """Sample the analytic background profile over the padded staggered meshes.

    The native registry owns profile evaluation. Normal components are sampled on
    their staggered faces and the out-of-plane component at cell centres.
    """
    bg = deck.background
    xc, yc, xf, yf, _dx, _dy = _padded_grids(deck.domain, nghost)
    params = dict(bg.params)
    if bg.profile == "uniform":
        params.update(bx0=float(bg.bx0), by0=float(bg.by0), bz0=float(bg.bz0))

    sampler = _core.mhd.sample_mhd_background_profile
    try:
        b0x = np.asarray(sampler(bg.profile, 0, xf, yc, params), dtype=np.float64)
        b0y = np.asarray(sampler(bg.profile, 1, xc, yf, params), dtype=np.float64)
        b0z = np.asarray(sampler(bg.profile, 2, xc, yc, params), dtype=np.float64)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"background_field.profile {bg.profile!r} could not be sampled: {exc}") \
            from None
    for name, arr in (("b0x", b0x), ("b0y", b0y), ("b0z", b0z)):
        if arr.shape != shape or not np.all(np.isfinite(arr)):
            raise ValueError(
                f"background_field.profile {bg.profile!r} produced invalid {name} "
                f"samples (shape {arr.shape}, expected {shape})")
    return b0x, b0y, b0z


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


def _set_primitive(state: dict, rho, vx, vy, vz, p, bx, by, bz, gamma) -> None:
    """Write conserved variables from primitive fields into ``state`` in place."""
    state["rho"] = rho
    state["mx"], state["my"], state["mz"] = mhd_num.momentum(rho, vx, vy, vz)
    state["bx"] = bx
    state["by"] = by
    state["bz"] = bz
    state["energy"] = mhd_num.primitive_to_energy(
        rho, p, vx, vy, vz, bx, by, bz, gamma)


def _ic_brio_wu(deck: MhdDeck, nghost: int) -> dict:
    """Brio-Wu shock tube: a 1D-in-x split state at ``interface``.

    params: interface (float), left/right each {rho,p,vx,vy,vz,bx,by,bz}.
    bx is the normal (continuous) face field, uniform across the interface; by is
    seeded into by_face; bz cell-centered. Convert primitive -> conserved.

    The two states are piecewise constant, so evaluating at element centers gives
    the exact element average everywhere except the single cut cell containing
    ``interface``. This is a discontinuous benchmark with no high-order continuum
    convergence rate; see the module docstring.
    """
    p = deck.initial.params
    gamma = deck.numerics.gamma
    interface = float(p.get("interface", 0.5))
    left = p["left"]
    right = p["right"]
    xc, yc, xf, yf, dx, dy = _padded_grids(deck.domain, nghost)
    shape = xc.shape
    state = _empty_state(shape)

    # Left for x < interface, right otherwise. Use cell-center x for the cell
    # quantities; bx is continuous (same on both sides for the standard problem)
    # so the face split matches the cell split.
    mask_left = xc < interface

    def split(key):
        return np.where(mask_left, float(left[key]), float(right[key]))

    rho = split("rho")
    pr = split("p")
    vx = split("vx")
    vy = split("vy")
    vz = split("vz")
    bx = split("bx")
    by = split("by")
    bz = split("bz")
    _set_primitive(state, rho, vx, vy, vz, pr, bx, by, bz, gamma)
    return _pack(state)


def _ic_alfven_wave(deck: MhdDeck, nghost: int) -> dict:
    """Circularly-polarized Alfven wave traveling along +x (exact CP eigenmode).

    params: rho, p, b0, amplitude (A), wavenumber (n, full wavelengths across
    lx), polarization ("circular").

    With total axial background B0 and Alfven speed |B0|/sqrt(rho), the exact CP
    eigen-relation (matching the example deck) is, for
    ``k = 2*pi*n/lx`` and ``xi=x-origin_x``::

        By =  A sin(k xi),    Bz =  A cos(k xi)
        vy = -sign(B0) A/sqrt(rho) sin(k xi)
        vz = -sign(B0) A/sqrt(rho) cos(k xi)

    so dv = -dB/sqrt(rho) (a +x-propagating Alfven wave). Bx = B0 (uniform),
    vx = 0, rho and p uniform.

    The native state is finite-volume data, so the transverse momentum and
    ``bz_cell`` entries are exact cell averages and ``by_face`` is the exact
    average over its x-directed face.  All therefore carry
    ``sinc(k*dx/2)`` relative to the value at the cell centre.  The exact total
    energy does *not* carry that factor: circular polarization makes the sum of
    the sine/cosine kinetic and magnetic energies spatially constant.  Preserve
    that sub-cell variance explicitly after forming the averaged primitives.
    """
    p = deck.initial.params
    gamma = deck.numerics.gamma
    rho0 = float(p.get("rho", 1.0))
    pr0 = float(p.get("p", 0.1))
    b0 = float(p.get("b0", 1.0))
    total_b0 = _alfven_reference_bx(deck)
    amp = float(p.get("amplitude", 1.0e-6))
    n = _as_exact_int(p.get("wavenumber", 1),
                      "initial.params.wavenumber")
    xc, yc, xf, yf, dx, dy = _padded_grids(deck.domain, nghost)
    shape = xc.shape
    state = _empty_state(shape)
    k = 2.0 * np.pi * n / deck.domain.lx_m
    phase = k * (xc - deck.domain.origin_x_m)
    # numpy.sinc(q) = sin(pi*q)/(pi*q).  Here k*dx/2 = pi*n/nx.
    fv_average = float(np.sinc(n / deck.domain.nx))
    # In SI the eigen-relation uses dB/sqrt(mu0*rho); normalized decks already
    # carry B in mu0=1 units.
    magnetic_velocity_scale = (1.0 / mhd_units.SQRT_MU0
                               if deck.units == "SI" else 1.0)
    inv_sqrt_rho = (math.copysign(magnetic_velocity_scale, total_b0)
                    / math.sqrt(rho0))

    rho = np.full(shape, rho0)
    vx = np.zeros(shape)
    # Exact volume averages of the transverse velocity perturbations.
    vy = -amp * inv_sqrt_rho * fv_average * np.sin(phase)
    vz = -amp * inv_sqrt_rho * fv_average * np.cos(phase)
    # bx is uniform (background), so the face value equals the cell value.
    bx = np.full(shape, b0)
    # A y-normal face spans one cell in x, so its x-only By profile has the same
    # sinc average as a cell volume.  Bz is cell-centred and volume-averaged.
    # div B remains zero because Bx is uniform and By is independent of y.
    by = amp * fv_average * np.sin(phase)
    bz = amp * fv_average * np.cos(phase)
    pr = np.full(shape, pr0)
    _set_primitive(state, rho, vx, vy, vz, pr, bx, by, bz, gamma)
    # _set_primitive sees averaged v and B and would therefore omit the
    # unresolved sin/cos variance.  In internal magnetic units the missing
    # kinetic and magnetic halves sum to this constant exact cell-average
    # contribution.  build_initial_state's later face-collocation adjustment
    # leaves this explicit sub-cell energy correction untouched.
    state["energy"] += ((amp * magnetic_velocity_scale) ** 2
                        * (1.0 - fv_average * fv_average))
    return _pack(state)


def _ic_orszag_tang(deck: MhdDeck, nghost: int) -> dict:
    """Orszag-Tang vortex in the uniformly rescaled ``mu0 = 1`` convention.

    params: b0. In domain-relative coordinates
    ``xi=(x-origin_x)/lx``, ``eta=(y-origin_y)/ly``::

        rho = gamma^2,  p = gamma
        v = (-sin(2 pi eta),  sin(2 pi xi),  0)
        B = b0 * (-sin(2 pi eta),  sin(4 pi xi),  0)

    The in-plane B is seeded at FACE locations (Bx on the left face -> uses
    cell-center y; By on the bottom face -> uses cell-center x) which is the
    standard cell-centered->face sampling for OT; with this analytic profile the
    discrete div B from the staggered seed is at round-off because Bx depends
    only on y and By only on x.

    .. note::

       This is a **midpoint projection, not an exact finite-volume one**. Every
       primitive is evaluated at its element center and stored in an
       average-valued field. Because the profile is smooth and nonlinear
       (sin/cos), the midpoint value differs from the true element moment by
       ``O(h^2)``. The resulting discrete state is a valid and standard OT
       benchmark, but initialization error alone limits a continuum convergence
       study to second order even under MP5/MP7. ``alfven_wave`` is the
       exactly-projected case; see the module docstring.
    """
    p = deck.initial.params
    gamma = deck.numerics.gamma
    b0 = float(p.get("b0", 1.0))
    xc, yc, xf, yf, dx, dy = _padded_grids(deck.domain, nghost)
    shape = xc.shape
    state = _empty_state(shape)
    xi = (xc - deck.domain.origin_x_m) / deck.domain.lx_m
    eta = (yc - deck.domain.origin_y_m) / deck.domain.ly_m

    rho = np.full(shape, gamma * gamma)
    pr = np.full(shape, gamma)
    vx = -np.sin(2.0 * np.pi * eta)
    vy = np.sin(2.0 * np.pi * xi)
    vz = np.zeros(shape)
    # Bx = -b0 sin(2 pi y): sample on the left face (x = xf) but it depends only
    # on y, so use cell-center y. By = b0 sin(4 pi x): depends only on x.
    bx = -b0 * np.sin(2.0 * np.pi * eta)
    by = b0 * np.sin(4.0 * np.pi * xi)
    bz = np.zeros(shape)
    _set_primitive(state, rho, vx, vy, vz, pr, bx, by, bz, gamma)
    return _pack(state)


def _ic_blast(deck: MhdDeck, nghost: int) -> dict:
    """Magnetized blast: high-pressure disk in a uniformly magnetized ambient.

    params: rho_ambient, p_ambient, p_core, r_in, center [x,y], bx, by, bz.
    Uniform B everywhere (face value = cell value); pressure is p_core inside
    r < r_in, p_ambient outside; density uniform = rho_ambient. v = 0.

    Every field is piecewise constant, so element-center evaluation is the exact
    element average except in the cells cut by the disk edge. Discontinuous
    benchmark; see the module docstring.
    """
    p = deck.initial.params
    gamma = deck.numerics.gamma
    rho_amb = float(p.get("rho_ambient", 1.0))
    p_amb = float(p.get("p_ambient", 0.1))
    p_core = float(p.get("p_core", 10.0))
    r_in = float(p.get("r_in", 0.1))
    center = p.get("center", [0.0, 0.0])
    cx, cy = float(center[0]), float(center[1])
    bx0 = float(p.get("bx", 0.0))
    by0 = float(p.get("by", 0.0))
    bz0 = float(p.get("bz", 0.0))

    xc, yc, xf, yf, dx, dy = _padded_grids(deck.domain, nghost)
    shape = xc.shape
    state = _empty_state(shape)
    r = np.hypot(xc - cx, yc - cy)

    rho = np.full(shape, rho_amb)
    pr = np.where(r < r_in, p_core, p_amb)
    vx = np.zeros(shape)
    vy = np.zeros(shape)
    vz = np.zeros(shape)
    bx = np.full(shape, bx0)
    by = np.full(shape, by0)
    bz = np.full(shape, bz0)
    _set_primitive(state, rho, vx, vy, vz, pr, bx, by, bz, gamma)
    return _pack(state)


def _ic_rotor(deck: MhdDeck, nghost: int) -> dict:
    """MHD rotor: dense rotating disk r<r0, linear taper r0..r1, ambient outside.

    params: center [x,y], r0, r1, rho_in, rho_out, u0 (rim speed at r0,
    omega = u0/r0), p, bx, by, bz.
    Inside the disk v = omega*(-(y-cy), (x-cx), 0); in the taper rho and the rim
    speed blend linearly (f = (r1-r)/(r1-r0)); ambient is static. Uniform p, B.

    Sampled at element centers. rho, p and B are piecewise constant and the disk
    velocity is linear in (x,y), so the midpoint value is the exact element
    average away from the r0/r1 taper edges; the radial taper and those edges are
    the only cells that differ from the true moment. Discontinuous benchmark; see
    the module docstring.
    """
    p = deck.initial.params
    gamma = deck.numerics.gamma
    center = p.get("center", [0.5, 0.5])
    cx, cy = float(center[0]), float(center[1])
    r0 = float(p.get("r0", 0.1))
    r1 = float(p.get("r1", 0.115))
    rho_in = float(p.get("rho_in", 10.0))
    rho_out = float(p.get("rho_out", 1.0))
    u0 = float(p.get("u0", 2.0))
    pr0 = float(p.get("p", 1.0))
    bx0 = float(p.get("bx", 0.0))
    by0 = float(p.get("by", 0.0))
    bz0 = float(p.get("bz", 0.0))
    omega = u0 / r0

    xc, yc, xf, yf, dx, dy = _padded_grids(deck.domain, nghost)
    shape = xc.shape
    state = _empty_state(shape)
    rx = xc - cx
    ry = yc - cy
    r = np.hypot(rx, ry)

    # Linear taper weight f: 1 inside (r<=r0), 0 outside (r>=r1).
    f = np.clip((r1 - r) / (r1 - r0), 0.0, 1.0)
    inside = r <= r0
    f = np.where(inside, 1.0, f)
    f = np.where(r >= r1, 0.0, f)

    rho = rho_out + f * (rho_in - rho_out)
    # Rigid-body rotation inside. In the transition use the standard rotor
    # profile v_phi=f*u0, continuous with the rim speed at r=r0 and tapering to
    # rest at r=r1. ``speed_factor`` is v_phi/r for the components below; the
    # taper never contains r=0.
    speed_factor = np.zeros_like(r)
    speed_factor[inside] = omega
    taper = (r > r0) & (r < r1)
    speed_factor[taper] = f[taper] * u0 / r[taper]
    vx = -speed_factor * ry
    vy = speed_factor * rx
    # Zero exactly outside the taper.
    vx = np.where(r >= r1, 0.0, vx)
    vy = np.where(r >= r1, 0.0, vy)
    vz = np.zeros(shape)
    pr = np.full(shape, pr0)
    bx = np.full(shape, bx0)
    by = np.full(shape, by0)
    bz = np.full(shape, bz0)
    _set_primitive(state, rho, vx, vy, vz, pr, bx, by, bz, gamma)
    return _pack(state)


def _ic_confined_blob(deck: MhdDeck, nghost: int) -> dict:
    """Confined plasma blob on a uniform out-of-plane (toroidal) field.

    A denser, higher-pressure plasma inside a centered square bore, ambient
    outside, initially at rest. The IN-PLANE perturbation field b is zero (so the
    constrained-transport div(b) starts and stays at exactly zero under any
    boundary closure); the confining POLOIDAL field is supplied separately as a
    static, non-uniform field-split background B0 (see ``background_field`` with a
    coil ``A`` file). The out-of-plane ``bz`` is a uniform toroidal guide field
    carried in the evolving state (uniform, so any open BC preserves it).

    params:
      bz        : uniform out-of-plane (toroidal) field (default 0.1).
      rho_in/out, p_in/out : confined-blob vs ambient density/pressure.
      blob_half : half-width (m) of the centered square plasma blob.

    Sampled at element centers. All fields are piecewise constant, so this is the
    exact element average except in the cells cut by the square bore edge.
    Discontinuous benchmark; see the module docstring.
    """
    p = deck.initial.params
    gamma = deck.numerics.gamma
    xc, yc, xf, yf, dx, dy = _padded_grids(deck.domain, nghost)
    shape = xc.shape
    state = _empty_state(shape)

    rho_in = float(p.get("rho_in", 10.0))
    rho_out = float(p.get("rho_out", 1.0))
    p_in = float(p.get("p_in", 1.0))
    p_out = float(p.get("p_out", 0.1))
    cx = deck.domain.origin_x_m + 0.5 * deck.domain.lx_m
    cy = deck.domain.origin_y_m + 0.5 * deck.domain.ly_m
    half = float(p.get("blob_half", 0.25 * min(deck.domain.lx_m,
                                               deck.domain.ly_m)))
    inside = (np.abs(xc - cx) <= half) & (np.abs(yc - cy) <= half)
    rho = np.where(inside, rho_in, rho_out)
    pr = np.where(inside, p_in, p_out)
    vx = np.zeros(shape)
    vy = np.zeros(shape)
    vz = np.zeros(shape)
    # In-plane perturbation b = 0; uniform toroidal guide field in the state.
    bx = np.zeros(shape)
    by = np.zeros(shape)
    bz = np.full(shape, float(p.get("bz", 0.1)))

    _set_primitive(state, rho, vx, vy, vz, pr, bx, by, bz, gamma)
    return _pack(state)
