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
from .._deck import (
    as_finite as _as_finite,
    require_finite as _require_finite,
    require_positive_finite as _require_positive_finite,
    parse_side_map as _parse_side_map,
)
from . import numerics as mhd_num


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
    ``file`` is given. ``params`` is a free dict for future named profiles.
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
    # so B0 is exactly divergence-free; the uniform out-of-plane bz0 is added as
    # the toroidal component. This is the one supported NON-UNIFORM background: a
    # curl-free (vacuum coil) field carries no net Maxwell self-force, so the
    # field-split conservative bookkeeping stays valid without static sources.
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

    def _validate_background(self) -> None:
        bg = self.background
        if not bg.enabled:
            return
        # The cylindrical geometric source builds its 1/r curvature terms from the
        # perturbation b only, while the radial flux uses the total field B = B0 + b;
        # combining a static background with cylindrical geometry would apply an
        # inconsistent update, so reject it (mirrors the C++ MhdSolver2D guard).
        if self.geometry == "cylindrical":
            raise ValueError(
                "background_field.enabled is not supported with geometry "
                "'cylindrical' (the geometric source does not fold in B0)")
        # The profile must be a live registered background-field profile so a
        # newly-registered profile needs no Python edit (mirrors the numerics
        # scheme validation).
        _check_registered(bg.profile,
                          _core.mhd.registered_mhd_background_profiles(),
                          "background_field.profile")
        if not isinstance(bg.params, dict):
            raise ValueError("background_field.params must be a mapping")
        # Uniform-vector parameters must be finite. (They are still parsed/stored
        # for a non-uniform profile, but only consumed by the uniform profile.)
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


def _parse_boundary(d: dict | None) -> BoundaryConfig:
    if d is None:
        return BoundaryConfig()
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
    file_raw = d.get("file")
    a_file_raw = d.get("a_file")
    return BackgroundConfig(
        enabled=bool(d.get("enabled", False)),
        profile=str(d.get("profile", "uniform")),
        bx0=float(d.get("bx0", 0.0)),
        by0=float(d.get("by0", 0.0)),
        bz0=float(d.get("bz0", 0.0)),
        params=dict(d.get("params", {}) or {}),
        file=None if file_raw is None else str(file_raw),
        a_file=None if a_file_raw is None else str(a_file_raw),
    )


def parse(data: dict) -> MhdDeck:
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
        data = yaml.safe_load(fh)
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
        "confined_blob": _ic_confined_blob,
    }
    builder = builders[deck.initial.type]
    return builder(deck, nghost)


# =============================================================================
# Static background magnetic field B0 (field-split form B = B0 + b)
# =============================================================================

# Round-off tolerance scale for the discrete divergence-free check on a seeded
# background field. The threshold is `_DIVB_TOL * max(1, |B0|_inf / min(dx,dy))`
# so a uniform/constant field passes trivially and a malformed (div-B != 0) file
# is rejected.
_DIVB_TOL = 1.0e-9


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
      Only "uniform" is registered today (so this reduces to the constants), but
      the sampling is written generically so a future profile slots in.
    * **file** (``file:`` given): ``np.load`` the npz and read arrays ``b0x``,
      ``b0y``, ``b0z`` each shaped ``(ny+2g, nx+2g)`` or flat ``(storage,)``;
      reshape/flatten to the 1-D storage layout.

    In ALL modes the interior discrete face-divergence of the assembled field is
    checked and a non-divergence-free background raises ``ValueError``.
    """
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

    # The coil A-file mode builds a NON-UNIFORM but curl-free B0 from a vector
    # potential; it is solenoidal by construction and skips the uniformity check
    # (justified below). Every other mode produces a uniform field.
    coil_mode = bg.a_file is not None
    if coil_mode:
        b0x, b0y, b0z = _background_from_a_file(deck, nghost, shape)
    elif bg.file is not None:
        b0x, b0y, b0z = _background_from_file(bg.file, deck.source_dir, shape,
                                              storage)
    else:
        b0x, b0y, b0z = _background_from_profile(deck, nghost, shape)

    # Divergence-free check (interior face divergence). Scale the tolerance by the
    # field magnitude so it survives a large but smooth uniform field while still
    # rejecting a genuinely non-solenoidal seed.
    inv_dmin = 1.0 / min(dx, dy)
    scale = max(1.0, float(np.max(np.abs(b0x)) + np.max(np.abs(b0y))) * inv_dmin)
    divb = mhd_num.background_divergence_linf(b0x, b0y, nx, ny, g, dx, dy)
    if divb > _DIVB_TOL * scale:
        raise ValueError(
            f"background_field is not divergence-free: max |div B0| = {divb:.3e} "
            f"exceeds tolerance {_DIVB_TOL * scale:.3e}. A uniform or staggered "
            f"solenoidal field is required.")

    # Uniformity check. The static field-split residual uses a conservative-flux-
    # only bookkeeping (the HLLD energy back-correction f_E_pert = f_E_tot - B0.F_B
    # and the total-field Maxwell stress). That is exact ONLY for a spatially
    # CONSTANT B0: a non-uniform background carries a magnetic-pressure gradient
    # grad(0.5|B0|^2) and tension, plus a grad(B0).F_b energy term, which would
    # need explicit static source terms in the momentum/energy residual to stay
    # conservative.
    #
    # EXCEPTION (coil A-file mode): a CURL-FREE vacuum coil field carries no net
    # Maxwell self-force -- div(B0 B0 - 0.5|B0|^2 I) = (curl B0) x B0 = 0 -- so the
    # missing static momentum source is exactly zero (not merely small) in the
    # continuum, and the energy cross-term grad(B0).F_b integrates to a boundary
    # flux for the smooth coil field. The split therefore stays conservative to
    # truncation order, so the uniformity guard is skipped for this mode. (It is a
    # vacuum field threading the bore, the intended physical use.)
    if not coil_mode:
        for name, comp in (("b0x", b0x), ("b0y", b0y), ("b0z", b0z)):
            cmin = float(np.min(comp))
            cmax = float(np.max(comp))
            spread = cmax - cmin
            tol = 1.0e-12 * max(1.0, abs(cmin), abs(cmax))
            if spread > tol:
                raise ValueError(
                    f"background_field component {name} is not spatially uniform "
                    f"(range [{cmin:.3e}, {cmax:.3e}]). Only a spatially-uniform "
                    f"background field is supported: the field-split solver carries "
                    f"no source terms for a non-uniform B0, so a varying background "
                    f"would violate energy/momentum conservation. Supply a constant "
                    f"vector (bx0/by0/bz0), a file whose arrays are constant per "
                    f"component, or a curl-free coil field via a_file.")

    return {"b0x": b0x.reshape(-1), "b0y": b0y.reshape(-1), "b0z": b0z.reshape(-1)}


def _background_from_a_file(deck: MhdDeck, nghost: int, shape):
    """Build a non-uniform, divergence-free B0 from a coil vector-potential npz.

    The coil CLI writes ``A_xyz_grid`` on the cell-corner grid of the FULL padded
    domain (so B0 is defined in the ghost layers too -- B0 is static and never
    ghost-refilled, and the reconstruction stencil reaches `nghost` cells past the
    interior). With the lab Y=0 slice and the mapping MHD-x = lab-X, MHD-y = lab-Z,
    out-of-plane = lab-Y, the saved array has shape ``(Ny+1, 1, Nx+1, 3)`` where
    ``Nx = nx + 2g``, ``Ny = ny + 2g`` are the padded cell counts. The in-plane B0
    is the discrete curl of the corner lab-Y component A[j, i]:

        b0x_face(i,j) = -(A[j+1,i] - A[j,i]) / dy      # B_R on the left face
        b0y_face(i,j) =  (A[j,i+1] - A[j,i]) / dx      # B_z on the bottom face

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
    if "A_xyz_grid" not in loaded:
        raise ValueError(
            f"background_field.a_file {file_rel!r} is missing array 'A_xyz_grid' "
            "(run the coil CLI with output.fields including 'A_xyz_grid')")
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
    a_ji = a_ji * b_scale

    b0x = -(a_ji[1:, :] - a_ji[:-1, :]) / dy  # (height, pitch+1) -> trim to faces
    b0y = (a_ji[:, 1:] - a_ji[:, :-1]) / dx   # (height+1, pitch) -> trim to faces
    # Face arrays are stored on the (height, pitch) cell layout: bx_face uses the
    # left face (drop the extra right column), by_face the bottom face (drop the
    # extra top row).
    b0x = b0x[:, :pitch].copy()
    b0y = b0y[:height, :].copy()
    b0z = np.full(shape, float(bg.bz0))
    return b0x, b0y, b0z


def _background_from_profile(deck: MhdDeck, nghost: int, shape):
    """Sample the analytic background profile over the padded staggered meshes.

    Only the "uniform" profile is registered today, so this reduces to constant
    components (b0x==bx0, b0y==by0, b0z==bz0). The face/cell mesh plumbing is
    written generically so a future spatially-varying profile (b0x at xf, b0y at
    yf, b0z at cell centers) slots in without changing the call site.
    """
    bg = deck.background
    xc, yc, xf, yf, dx, dy = _padded_grids(deck.domain, nghost)
    _ = (xc, yc, xf, yf)  # staggered meshes for a future spatial profile.
    if bg.profile == "uniform":
        b0x = np.full(shape, float(bg.bx0))
        b0y = np.full(shape, float(bg.by0))
        b0z = np.full(shape, float(bg.bz0))
        return b0x, b0y, b0z
    # The profile name validated against the live C++ registry, but only "uniform"
    # has a Python-side analytic sampler today. A different registered profile has
    # no host sampler to fill the staggered B0 buffers, so fail loudly rather than
    # silently substitute a constant field (which would misrepresent the requested
    # profile). A future spatial profile adds its sampler here using the xf/yf/xc
    # meshes above.
    raise NotImplementedError(
        f"background_field.profile {bg.profile!r} is registered but has no "
        f"Python-side sampler yet; only 'uniform' is supported from the deck. "
        f"Supply uniform components (bx0/by0/bz0) or load B0 from a file.")


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
    out = []
    for name in ("b0x", "b0y", "b0z"):
        if name not in loaded:
            raise ValueError(
                f"background_field.file {file_rel!r} is missing array {name!r} "
                f"(needs b0x, b0y, b0z)")
        arr = np.asarray(loaded[name], dtype=np.float64)
        if arr.shape == shape:
            out.append(arr.copy())
        elif arr.shape == (storage,):
            out.append(arr.reshape(shape))
        else:
            raise ValueError(
                f"background_field.file {file_rel!r} array {name!r} has shape "
                f"{arr.shape}; expected {shape} or ({storage},)")
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
