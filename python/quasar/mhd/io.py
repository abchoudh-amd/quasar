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
``{brio_wu, alfven_wave, orszag_tang, blast, rotor}``; ``initial.params`` is a
free dict passed to the matching IC generator below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import yaml

from .. import _core
from .._deck import require as _require
from . import numerics as mhd_num


# Sanity ceilings on deck-supplied sizes that flow into device allocations
# (mirrors quasar.pic.io). Guard against typos / hostile decks, not legitimate
# large runs.
MAX_GRID_DIM = 1 << 16        # 65536 cells per axis
MAX_GRID_CELLS = 1 << 30      # ~1.07e9 cells total

# Canonical initial-condition tokens (the SINGLE source of truth, mirrored by the
# example decks). Validation rejects any initial.type not in this set.
INITIAL_TYPES = ("brio_wu", "alfven_wave", "orszag_tang", "blast", "rotor")

# Conserved-state components written to / read from the solver and the .npz.
STATE_COMPONENTS = ("rho", "mx", "my", "mz", "energy", "bx", "by", "bz")


def _as_finite(value: float, context: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{context} must be a finite number") from None
    if not math.isfinite(v):
        raise ValueError(f"{context} must be finite")
    return v


def _require_finite(value: float, context: str) -> None:
    _as_finite(value, context)


def _require_positive_finite(value: float, context: str) -> None:
    if _as_finite(value, context) <= 0:
        raise ValueError(f"{context} must be positive")


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

    ``fluid`` and ``field`` are independent (a reflecting wall imposes different
    symmetries on momentum than on the magnetic field). Names are validated
    against the live C++ registries.
    """
    fluid: tuple[str, str, str, str] = (
        "periodic", "periodic", "periodic", "periodic")
    field: tuple[str, str, str, str] = (
        "periodic", "periodic", "periodic", "periodic")


@dataclass
class MhdDeck:
    domain: Domain
    numerics: Numerics = field(default_factory=Numerics)
    initial: Union[Initial, None] = None
    time: Time = field(default_factory=Time)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    units: str = "normalized"
    geometry: str = "cartesian"
    raw: dict = field(default_factory=dict)

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
        if self.geometry == "cylindrical":
            self._validate_cylindrical()

    def _validate_domain(self) -> None:
        if self.domain.nx <= 0 or self.domain.ny <= 0:
            raise ValueError("domain.nx and domain.ny must be positive")
        if self.domain.nx > MAX_GRID_DIM or self.domain.ny > MAX_GRID_DIM:
            raise ValueError(
                f"domain.nx/ny must be <= {MAX_GRID_DIM} (got "
                f"{self.domain.nx}x{self.domain.ny})")
        if self.domain.nx * self.domain.ny > MAX_GRID_CELLS:
            raise ValueError(f"domain.nx*ny must be <= {MAX_GRID_CELLS} cells")
        _require_positive_finite(self.domain.lx_m, "domain.lx_m")
        _require_positive_finite(self.domain.ly_m, "domain.ly_m")
        _require_finite(self.domain.origin_x_m, "domain.origin_x_m")
        _require_finite(self.domain.origin_y_m, "domain.origin_y_m")

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

    def _validate_time(self) -> None:
        if isinstance(self.time.dt_s, str) and self.time.dt_s != "auto":
            raise ValueError("time.dt_s must be a float or the string 'auto'")
        if not isinstance(self.time.dt_s, str):
            _require_positive_finite(self.time.dt_s, "time.dt_s")
        if self.time.steps <= 0:
            raise ValueError("time.steps must be positive")
        if self.time.t_end is not None:
            _require_positive_finite(self.time.t_end, "time.t_end")

    def _validate_diagnostics(self) -> None:
        if self.diagnostics.cadence < 0:
            raise ValueError("diagnostics.cadence must be >= 0")
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

    def _validate_cylindrical(self) -> None:
        # The on-axis (r=0) closure assumes the radial domain starts exactly at
        # r=0; finite inner radius / annular domains are not supported (mirrors
        # the PIC cylindrical validation).
        if self.domain.origin_x_m != 0.0:
            raise ValueError(
                "geometry 'cylindrical': domain.origin_x_m must be 0 (the m=0 "
                "on-axis scheme requires the radial domain to start at r=0; "
                "finite inner radius / annular domains are not supported yet)")


def _check_registered(name: str, allowed: Sequence[str], context: str) -> None:
    if name not in set(allowed):
        raise ValueError(
            f"{context} {name!r} must be one of {sorted(set(allowed))}")


def _parse_domain(d: dict) -> Domain:
    return Domain(
        nx=int(_require(d, "nx", "domain")),
        ny=int(_require(d, "ny", "domain")),
        lx_m=float(_require(d, "lx_m", "domain")),
        ly_m=float(_require(d, "ly_m", "domain")),
        origin_x_m=float(d.get("origin_x_m", 0.0)),
        origin_y_m=float(d.get("origin_y_m", 0.0)),
    )


def _parse_numerics(d: dict | None) -> Numerics:
    if d is None:
        return Numerics()
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
    return Initial(
        type=str(_require(d, "type", "initial")),
        params=dict(d.get("params", {}) or {}),
    )


def _parse_time(d: dict | None) -> Time:
    if d is None:
        return Time()
    dt_raw = d.get("dt_s", "auto")
    dt_s: Union[float, str] = dt_raw if isinstance(dt_raw, str) else float(dt_raw)
    t_end = d.get("t_end")
    return Time(dt_s=dt_s, steps=int(d.get("steps", 100)),
                t_end=None if t_end is None else float(t_end))


def _parse_diagnostics(d: dict | None) -> Diagnostics:
    if d is None:
        return Diagnostics()
    return Diagnostics(
        output_path=str(d.get("output_path", "out.npz")),
        cadence=int(d.get("cadence", 0)),
        fields=list(d.get("fields", list(STATE_COMPONENTS))),
        divb=bool(d.get("divb", True)),
    )


# Side ordering and the scalar | 4-list | side-keyed-map flexibility are shared
# with the PIC deck (quasar.pic.io._parse_side_map / _SIDE_KEYS); duplicated here
# (small, no cross-package import) so the MHD loader stays self-contained.
_SIDE_KEYS = ("x_lo", "x_hi", "y_lo", "y_hi")


def _parse_side_map(spec, default: str, what: str) -> tuple[str, str, str, str]:
    # Accepts a scalar (all sides), a 4-element list ([x_lo, x_hi, y_lo, y_hi]),
    # or a dict keyed by side name.
    if spec is None:
        return (default, default, default, default)
    if isinstance(spec, str):
        return (spec, spec, spec, spec)
    if isinstance(spec, (list, tuple)) and len(spec) == 4:
        return (str(spec[0]), str(spec[1]), str(spec[2]), str(spec[3]))
    if isinstance(spec, dict):
        return tuple(str(spec.get(k, default)) for k in _SIDE_KEYS)  # type: ignore[return-value]
    raise ValueError(f"{what} must be a string, 4-element list, or side-keyed map")


def _parse_boundary(d: dict | None) -> BoundaryConfig:
    if d is None:
        return BoundaryConfig()
    return BoundaryConfig(
        fluid=_parse_side_map(d.get("fluid"), "periodic", "boundary.fluid"),
        field=_parse_side_map(d.get("field"), "periodic", "boundary.field"),
    )


def parse(data: dict) -> MhdDeck:
    deck = MhdDeck(
        domain=_parse_domain(_require(data, "domain", "deck")),
        numerics=_parse_numerics(data.get("numerics")),
        initial=_parse_initial(data.get("initial")),
        time=_parse_time(data.get("time")),
        diagnostics=_parse_diagnostics(data.get("diagnostics")),
        boundary=_parse_boundary(data.get("boundary")),
        units=str(data.get("units", "normalized")),
        geometry=str(data.get("geometry", "cartesian")),
        raw=data,
    )
    deck.validate()
    return deck


def load(path: Union[str, Path]) -> MhdDeck:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return parse(data)


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
    g = nghost
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
    return xc, yc, xf, yf, dx, dy


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
    builders = {
        "brio_wu": _ic_brio_wu,
        "alfven_wave": _ic_alfven_wave,
        "orszag_tang": _ic_orszag_tang,
        "blast": _ic_blast,
        "rotor": _ic_rotor,
    }
    builder = builders[deck.initial.type]
    return builder(deck, nghost)


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

    With background B0 along x and Alfven speed vA = B0/sqrt(rho), the exact CP
    eigen-relation (matching the example deck) is, for k = 2*pi*n/lx::

        By =  A sin(k x),    Bz =  A cos(k x)
        vy = -A/sqrt(rho) sin(k x) = -(A/B0) vA sin(k x)
        vz = -A/sqrt(rho) cos(k x)

    so dv = -dB/sqrt(rho) (a +x-propagating Alfven wave). Bx = B0 (uniform),
    vx = 0, rho and p uniform. Energy from primitive_to_energy.
    """
    p = deck.initial.params
    gamma = deck.numerics.gamma
    rho0 = float(p.get("rho", 1.0))
    pr0 = float(p.get("p", 0.1))
    b0 = float(p.get("b0", 1.0))
    amp = float(p.get("amplitude", 1.0e-6))
    n = int(p.get("wavenumber", 1))
    polarization = str(p.get("polarization", "circular"))
    if polarization != "circular":
        raise ValueError(
            f"alfven_wave: only 'circular' polarization is supported, got "
            f"{polarization!r}")

    xc, yc, xf, yf, dx, dy = _padded_grids(deck.domain, nghost)
    shape = xc.shape
    state = _empty_state(shape)
    k = 2.0 * np.pi * n / deck.domain.lx_m
    inv_sqrt_rho = 1.0 / math.sqrt(rho0)

    rho = np.full(shape, rho0)
    vx = np.zeros(shape)
    # Transverse velocity perturbations evaluated at cell centers.
    vy = -amp * inv_sqrt_rho * np.sin(k * xc)
    vz = -amp * inv_sqrt_rho * np.cos(k * xc)
    # bx is uniform (background), so the face value equals the cell value.
    bx = np.full(shape, b0)
    # by varies only with x; sample by_face at the bottom face -> same x as cell
    # center is fine for an x-only field. Use cell-center x for consistency with
    # the velocity sampling (div B stays ~0 because Bx is uniform and
    # d(By)/dy = 0).
    by = amp * np.sin(k * xc)
    bz = amp * np.cos(k * xc)
    pr = np.full(shape, pr0)
    _set_primitive(state, rho, vx, vy, vz, pr, bx, by, bz, gamma)
    return _pack(state)


def _ic_orszag_tang(deck: MhdDeck, nghost: int) -> dict:
    """Orszag-Tang vortex (gamma-derived ambient state, b0 = 1/sqrt(4 pi)).

    params: b0. On [0,1]^2::

        rho = gamma^2,  p = gamma
        v = (-sin(2 pi y),  sin(2 pi x),  0)
        B = b0 * (-sin(2 pi y),  sin(4 pi x),  0)

    The in-plane B is seeded at FACE locations (Bx on the left face -> uses
    cell-center y; By on the bottom face -> uses cell-center x) which is the
    standard cell-centered->face sampling for OT; with this analytic profile the
    discrete div B from the staggered seed is at round-off because Bx depends
    only on y and By only on x.
    """
    p = deck.initial.params
    gamma = deck.numerics.gamma
    b0 = float(p.get("b0", 1.0 / math.sqrt(4.0 * math.pi)))
    xc, yc, xf, yf, dx, dy = _padded_grids(deck.domain, nghost)
    shape = xc.shape
    state = _empty_state(shape)

    rho = np.full(shape, gamma * gamma)
    pr = np.full(shape, gamma)
    vx = -np.sin(2.0 * np.pi * yc)
    vy = np.sin(2.0 * np.pi * xc)
    vz = np.zeros(shape)
    # Bx = -b0 sin(2 pi y): sample on the left face (x = xf) but it depends only
    # on y, so use cell-center y. By = b0 sin(4 pi x): depends only on x.
    bx = -b0 * np.sin(2.0 * np.pi * yc)
    by = b0 * np.sin(4.0 * np.pi * xc)
    bz = np.zeros(shape)
    _set_primitive(state, rho, vx, vy, vz, pr, bx, by, bz, gamma)
    return _pack(state)


def _ic_blast(deck: MhdDeck, nghost: int) -> dict:
    """Magnetized blast: high-pressure disk in a uniformly magnetized ambient.

    params: rho_ambient, p_ambient, p_core, r_in, center [x,y], bx, by, bz.
    Uniform B everywhere (face value = cell value); pressure is p_core inside
    r < r_in, p_ambient outside; density uniform = rho_ambient. v = 0.
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
    r = np.sqrt((xc - cx) ** 2 + (yc - cy) ** 2)

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
    r = np.sqrt(rx * rx + ry * ry)
    # Avoid divide-by-zero at r=0 for the taper blend direction (v=0 there).
    r_safe = np.where(r == 0.0, 1.0, r)

    # Linear taper weight f: 1 inside (r<=r0), 0 outside (r>=r1).
    f = np.clip((r1 - r) / (r1 - r0), 0.0, 1.0)
    inside = r <= r0
    f = np.where(inside, 1.0, f)
    f = np.where(r >= r1, 0.0, f)

    rho = rho_out + f * (rho_in - rho_out)
    # Rigid-body rotation inside; in the taper scale the rim speed by f. The rim
    # speed at radius r for solid-body omega is omega*r; the taper damps it by f.
    speed_factor = np.where(inside, omega, f * omega)
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
    _ = r_safe  # documented guard; speed uses rx/ry directly so no division.
    _set_primitive(state, rho, vx, vy, vz, pr, bx, by, bz, gamma)
    return _pack(state)
