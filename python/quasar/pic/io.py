"""YAML schema for PIC input decks.

The C++ solver owns the numerical kernels; this module models the user-facing
deck as a small set of dataclasses, parses YAML into them, and validates.
The CLI (``quasar.pic.cli``) consumes the parsed deck.

Top-level structure::

    units: SI
    normalization: {reference_density_per_m3, reference_species}
    domain: {nx, ny, lx_m, ly_m, origin_x_m?, origin_y_m?}
    numerics: {fdtd_order, shape, current_filter?}
    external_field:
      evaluator:
        type: biot_savart
        conductors: [...]            # same schema as quasar.coil
    species:
      - name, charge_C, mass_kg, n_particles, initial: {distribution,
          density_per_m3, temperature_eV, drift_v?}
    time: {dt_s, steps}
    diagnostics: {output_path, cadence, fields, per_species}

Note: the PIC deck groups output under ``diagnostics.output_path`` (with cadence /
fields / per_species) because it writes a time series; the coil deck writes a
single snapshot and uses ``output.path`` instead. The two output schemas are
intentionally distinct (see ``quasar.coil.io``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

import yaml

from .. import _core
from .._deck import require as _require, triple as _triple
from .._deck import validate_evaluator_type as _validate_evaluator_type


# Sanity ceilings on deck-supplied sizes that flow into device allocations.
# These guard against typos / hostile decks requesting absurd buffers, not
# against legitimate large runs (raise them if a real workload needs more).
MAX_GRID_DIM = 1 << 16        # 65536 cells per axis
MAX_GRID_CELLS = 1 << 30      # ~1.07e9 cells total
MAX_PARTICLES = 1 << 31       # ~2.1e9 particles per species
# Canonical list of Yee field component names. The C++ yee_component_buffer in
# bindings/python/bind_pic.cpp and the E/B split in quasar.pic._units must mirror
# this set.
FIELD_COMPONENTS = ("ex", "ey", "ez", "bx", "by", "bz")


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


def _require_nonnegative_finite(value: float, context: str) -> None:
    if _as_finite(value, context) < 0:
        raise ValueError(f"{context} must be >= 0")


def _require_vec_finite(values: Sequence[float], context: str) -> None:
    for i, value in enumerate(values):
        _require_finite(value, f"{context}[{i}]")


def _vec3(xyz: Sequence[float] | None,
          default: tuple[float, float, float] = (0.0, 0.0, 0.0),
          ) -> tuple[float, float, float]:
    if xyz is None:
        return default
    return _triple(xyz)


def _matrix3(rows: Sequence[Sequence[float]] | None,
             ) -> tuple[tuple[float, float, float],
                        tuple[float, float, float],
                        tuple[float, float, float]] | None:
    if rows is None:
        return None
    if len(rows) != 3:
        raise ValueError("expected 3x3 matrix")
    return (_triple(rows[0]), _triple(rows[1]), _triple(rows[2]))


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
    fdtd_order: int = 2
    shape: str = "cic"
    current_filter: list[dict] = field(default_factory=list)


@dataclass
class Normalization:
    reference_density_per_m3: float = 1.0e18
    reference_species: str = "electron"


@dataclass
class ExternalField:
    evaluator_type: str
    conductors: list[dict] = field(default_factory=list)
    # Evaluator parameters are in SI for SI decks and internal units for
    # normalized decks. The sampler handles the conversion into solver units.
    uniform_b: tuple[float, float, float] = (0.0, 0.0, 0.0)
    uniform_e: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dipole_moment: tuple[float, float, float] | None = None
    dipole_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gradient_b0: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gradient_matrix: tuple[tuple[float, float, float],
                           tuple[float, float, float],
                           tuple[float, float, float]] | None = None
    gradient_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def evaluator_params(self) -> dict[str, list[float]]:
        """Deck parameters as the ``name -> flat float list`` map the C++
        ``IFieldEvaluator.configure`` seam consumes (Vec3 = 3 elements, Mat3x3 = 9
        row-major). Every evaluator reads only the keys it knows, so this returns
        the union for all analytic types; biot_savart ignores all of them.

        ``origin`` is shared by the dipole and gradient evaluators (only one is
        ever active for a given deck), so a single key carries it.
        """
        params: dict[str, list[float]] = {
            "b0": list(self.uniform_b),
            "e0": list(self.uniform_e),
            "origin": list(self.dipole_origin),
        }
        if self.dipole_moment is not None:
            params["moment"] = list(self.dipole_moment)
        if self.gradient_matrix is not None:
            params["b0"] = list(self.gradient_b0)
            params["origin"] = list(self.gradient_origin)
            params["grad"] = [v for row in self.gradient_matrix for v in row]
        return params


@dataclass
class SpeciesInitial:
    distribution: str = "maxwellian_uniform"
    density_per_m3: float = 1.0e18
    temperature_eV: float = 1.0
    drift_v: tuple[float, float, float] = (0.0, 0.0, 0.0)
    region_x_min_m: float | None = None
    region_x_max_m: float | None = None
    region_y_min_m: float | None = None
    region_y_max_m: float | None = None


@dataclass
class Species:
    name: str
    charge_C: float
    mass_kg: float
    n_particles: int
    initial: SpeciesInitial


@dataclass
class Time:
    dt_s: Union[float, str] = "auto"
    steps: int = 100


@dataclass
class BoundaryConfig:
    """Per-side boundary kinds. Order: [x_min, x_max, y_min, y_max].

    ``particle`` is one of ``periodic`` (no-op), ``specular`` (reflect), or
    ``absorbing`` (mark out-of-domain particles as dead). ``field`` is one of
    ``periodic`` or ``pec`` (perfect electric conductor / reflecting wall).
    """
    particle: tuple[str, str, str, str] = (
        "periodic", "periodic", "periodic", "periodic")
    field: tuple[str, str, str, str] = (
        "periodic", "periodic", "periodic", "periodic")


@dataclass
class FieldsInitial:
    """Optional initial field seeding (for field-driven validation decks).

    * ``seed_perturbation``: a single-mode sinusoid in one E component
      (``component``, ``amplitude``, ``mode`` half-wavelengths along x).
    * ``seed_em_wave``: a propagating plane wave (``component`` Ez or Ey with its
      partner B set for +x/+y propagation, ``mode`` = (mx, my)).
    """
    type: str
    component: str = "Ex"
    amplitude: float = 1.0e-4
    mode: tuple[int, int] = (1, 0)


@dataclass
class Fields:
    initial: Union[FieldsInitial, None] = None


@dataclass
class Diagnostics:
    output_path: str = "out.npz"
    cadence: int = 0
    fields: list[str] = field(default_factory=lambda: ["bz"])
    per_species: bool = True

    def __post_init__(self) -> None:
        # Normalize component names once, so YAML-parsed and directly-constructed
        # decks agree before validate() checks membership in FIELD_COMPONENTS.
        self.fields = [str(name).lower() for name in self.fields]


@dataclass
class PicDeck:
    domain: Domain
    numerics: Numerics = field(default_factory=Numerics)
    normalization: Normalization = field(default_factory=Normalization)
    external_field: Union[ExternalField, None] = None
    species: list[Species] = field(default_factory=list)
    time: Time = field(default_factory=Time)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    fields: Fields = field(default_factory=Fields)
    units: str = "SI"
    # Which lab plane the 2D grid represents. "xy" (default) samples the external
    # field at lab z=0 with grid axes (x,y); "xz" samples the lab y=0 meridional
    # plane with grid axes (x,z) and the out-of-plane component along lab y.
    plane: str = "xy"
    # Coordinate system of the 2D grid. "cartesian" (default) treats axes as
    # (x, y) in the chosen plane; "cylindrical" treats them as axisymmetric r-z
    # with i=r (from nx / lx_m / origin_x_m) and j=z. The C++ solver applies the
    # on-axis closure for the inner-radius (x_lo) boundary.
    geometry: str = "cartesian"
    raw: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.units not in ("SI", "normalized"):
            raise ValueError("units must be 'SI' or 'normalized'")
        if self.plane not in ("xy", "xz"):
            raise ValueError("plane must be 'xy' or 'xz'")
        if self.geometry not in ("cartesian", "cylindrical"):
            raise ValueError(
                "geometry must be 'cartesian' or 'cylindrical'")
        self._validate_domain()
        self._validate_numerics()
        self._validate_external_field()
        # A deck must drive *something*: particles, an external field, or an
        # initial field seed. Field-only decks (e.g. EM-wave propagation, coil
        # confinement) legitimately have no species.
        if (not self.species and self.external_field is None
                and self.fields.initial is None):
            raise ValueError(
                "deck must define at least one of: species, external_field, "
                "or fields.initial")
        self._validate_species()
        if self.fields.initial is not None:
            _require_finite(self.fields.initial.amplitude, "fields.initial.amplitude")
        self._validate_time()
        self._validate_diagnostics()
        self._validate_boundary()
        # Run cylindrical cross-cutting checks last: they read already-validated
        # numerics/boundary fields, so any structural error in those surfaces its
        # own (more fundamental) message first.
        if self.geometry == "cylindrical":
            self._validate_cylindrical()

    def _validate_cylindrical(self) -> None:
        # i=r runs from origin_x_m. The m=0 on-axis scheme (axis BC auto-wiring,
        # the i==0 Er/Ephi pin in the FDTD curls, and the J0(j0n r/R) seed) all
        # assume the radial domain starts exactly at r=0, so require origin_x_m==0.
        # Finite inner radius / annular domains are not yet supported.
        if self.domain.origin_x_m != 0.0:
            raise ValueError(
                "geometry 'cylindrical': domain.origin_x_m must be 0 (the m=0 "
                "on-axis scheme requires the radial domain to start at r=0; "
                "finite inner radius / annular domains are not supported yet)")
        # The on-axis closure is only derived for the 2nd-order Yee curl.
        if self.numerics.fdtd_order != 2:
            raise ValueError(
                "geometry 'cylindrical': numerics.fdtd_order must be 2 "
                "(order-4 cylindrical axis closure is not supported yet)")
        # A periodic OUTER-radius (x_hi, side index 1) wall is unphysical for an
        # axisymmetric domain. x_lo (inner radius) is deliberately left alone:
        # BoundaryConfig defaults all sides to 'periodic' and the C++ solver
        # auto-replaces x_lo with the axis condition, so rejecting periodic x_lo
        # would break every normal cylindrical deck.
        if self.boundary.field[1] == "periodic":
            raise ValueError(
                "geometry 'cylindrical': boundary.field x_hi (outer radius) "
                "must not be 'periodic'")
        if self.boundary.particle[1] == "periodic":
            raise ValueError(
                "geometry 'cylindrical': boundary.particle x_hi (outer radius) "
                "must not be 'periodic'")

    def _validate_domain(self) -> None:
        if self.domain.nx <= 0 or self.domain.ny <= 0:
            raise ValueError("domain.nx and domain.ny must be positive")
        # Upper-bound grid and particle counts so a typo (or hostile deck) cannot
        # request a multi-terabyte device allocation; these flow straight into
        # Grid2D storage and per-species particle buffers.
        if self.domain.nx > MAX_GRID_DIM or self.domain.ny > MAX_GRID_DIM:
            raise ValueError(
                f"domain.nx/ny must be <= {MAX_GRID_DIM} (got "
                f"{self.domain.nx}x{self.domain.ny})")
        if self.domain.nx * self.domain.ny > MAX_GRID_CELLS:
            raise ValueError(
                f"domain.nx*ny must be <= {MAX_GRID_CELLS} cells")
        _require_positive_finite(self.domain.lx_m, "domain.lx_m")
        _require_positive_finite(self.domain.ly_m, "domain.ly_m")
        _require_finite(self.domain.origin_x_m, "domain.origin_x_m")
        _require_finite(self.domain.origin_y_m, "domain.origin_y_m")
        _require_positive_finite(
            self.normalization.reference_density_per_m3,
            "normalization.reference_density_per_m3")

    def _validate_numerics(self) -> None:
        if self.numerics.fdtd_order not in (2, 4):
            raise ValueError("numerics.fdtd_order must be 2 or 4")
        if self.numerics.shape not in ("cic", "tsc"):
            raise ValueError("numerics.shape must be 'cic' or 'tsc'")
        allowed_filters = set(_core.pic.registered_current_filters())
        for i, spec in enumerate(self.numerics.current_filter):
            if not isinstance(spec, dict):
                raise ValueError(f"numerics.current_filter[{i}] must be a mapping")
            name = spec.get("type")
            if name not in allowed_filters:
                raise ValueError(
                    f"numerics.current_filter[{i}].type {name!r} must be one of "
                    f"{sorted(allowed_filters)}")
            passes = int(spec.get("n_passes", spec.get("passes", 1)))
            if passes < 1:
                raise ValueError(
                    f"numerics.current_filter[{i}] passes must be >= 1")

    def _validate_external_field(self) -> None:
        if self.external_field is None:
            return
        ev = self.external_field.evaluator_type
        _validate_evaluator_type(ev, "external_field.evaluator.type")
        _require_vec_finite(
            self.external_field.uniform_b, "external_field.evaluator.B")
        _require_vec_finite(
            self.external_field.uniform_e, "external_field.evaluator.E")
        _require_vec_finite(
            self.external_field.dipole_origin,
            "external_field.evaluator.origin")
        _require_vec_finite(
            self.external_field.gradient_b0,
            "external_field.evaluator.B0")
        _require_vec_finite(
            self.external_field.gradient_origin,
            "external_field.evaluator.gradient_origin")
        if self.external_field.dipole_moment is not None:
            _require_vec_finite(
                self.external_field.dipole_moment,
                "external_field.evaluator.moment")
        if self.external_field.gradient_matrix is not None:
            for r, row in enumerate(self.external_field.gradient_matrix):
                _require_vec_finite(
                    row, f"external_field.evaluator.grad[{r}]")
        # Biot-Savart is driven by conductor geometry; the others are
        # closed-form and need no conductors.
        if ev == "biot_savart" and not self.external_field.conductors:
            raise ValueError("external_field.evaluator.conductors must be non-empty")
        if ev == "dipole" and self.external_field.dipole_moment is None:
            raise ValueError(
                "external_field.evaluator.moment_Am2 is required for type 'dipole'")
        if ev == "gradient" and self.external_field.gradient_matrix is None:
            raise ValueError(
                "external_field.evaluator.grad_T_per_m is required for type 'gradient'")

    def _validate_species(self) -> None:
        for sp in self.species:
            _require_finite(sp.charge_C, f"species {sp.name!r}: charge_C")
            _require_positive_finite(sp.mass_kg, f"species {sp.name!r}: mass_kg")
            if sp.n_particles <= 0:
                raise ValueError(f"species {sp.name!r}: n_particles must be positive")
            if sp.n_particles > MAX_PARTICLES:
                raise ValueError(
                    f"species {sp.name!r}: n_particles must be <= {MAX_PARTICLES} "
                    f"(got {sp.n_particles})")
            if sp.initial.distribution not in ("maxwellian_uniform",
                                                 "maxwellian_block"):
                raise ValueError(
                    f"species {sp.name!r}: initial.distribution "
                    f"{sp.initial.distribution!r} is not supported (only "
                    "'maxwellian_uniform' and 'maxwellian_block')"
                )
            if sp.initial.distribution == "maxwellian_block":
                bounds = (sp.initial.region_x_min_m, sp.initial.region_x_max_m,
                          sp.initial.region_y_min_m, sp.initial.region_y_max_m)
                if any(b is None for b in bounds):
                    raise ValueError(
                        f"species {sp.name!r}: maxwellian_block requires "
                        "initial.region.{x_min_m,x_max_m,y_min_m,y_max_m}"
                    )
                if sp.initial.region_x_min_m >= sp.initial.region_x_max_m:
                    raise ValueError(
                        f"species {sp.name!r}: region x_min_m must be < x_max_m")
                if sp.initial.region_y_min_m >= sp.initial.region_y_max_m:
                    raise ValueError(
                        f"species {sp.name!r}: region y_min_m must be < y_max_m")
                for label, value in zip(
                    ("x_min_m", "x_max_m", "y_min_m", "y_max_m"), bounds):
                    _require_finite(
                        value, f"species {sp.name!r}: region {label}")
            _require_nonnegative_finite(
                sp.initial.density_per_m3,
                f"species {sp.name!r}: density_per_m3")
            _require_nonnegative_finite(
                sp.initial.temperature_eV,
                f"species {sp.name!r}: temperature_eV")
            _require_vec_finite(sp.initial.drift_v, f"species {sp.name!r}: drift_v")

    def _validate_time(self) -> None:
        if isinstance(self.time.dt_s, str) and self.time.dt_s != "auto":
            raise ValueError("time.dt_s must be a float or the string 'auto'")
        if not isinstance(self.time.dt_s, str):
            _require_positive_finite(self.time.dt_s, "time.dt_s")
        if self.time.steps <= 0:
            raise ValueError("time.steps must be positive")

    def _validate_diagnostics(self) -> None:
        if self.diagnostics.cadence < 0:
            raise ValueError("diagnostics.cadence must be >= 0")
        for field_name in self.diagnostics.fields:
            if field_name not in FIELD_COMPONENTS:
                raise ValueError(
                    f"diagnostics.fields entry {field_name!r} must be one of "
                    f"{list(FIELD_COMPONENTS)}")

    def _validate_boundary(self) -> None:
        # Validate against the live C++ registry so a newly-registered boundary
        # needs no Python edit (the registry is the single source of truth).
        allowed_pbc = set(_core.pic.registered_particle_boundaries())
        for i, bc in enumerate(self.boundary.particle):
            if bc not in allowed_pbc:
                raise ValueError(
                    f"boundary.particle[{i}] = {bc!r} must be one of {sorted(allowed_pbc)}")
        allowed_fbc = set(_core.pic.registered_field_boundaries())
        for i, bc in enumerate(self.boundary.field):
            if bc not in allowed_fbc:
                raise ValueError(
                    f"boundary.field[{i}] = {bc!r} must be one of {sorted(allowed_fbc)}")


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
        fdtd_order=int(d.get("fdtd_order", 2)),
        shape=str(d.get("shape", "cic")),
        current_filter=list(d.get("current_filter", [])),
    )


def _parse_normalization(d: dict | None) -> Normalization:
    if d is None:
        return Normalization()
    return Normalization(
        reference_density_per_m3=float(d.get("reference_density_per_m3", 1.0e18)),
        reference_species=str(d.get("reference_species", "electron")),
    )


# --- Physical-component -> storage-slot translation (cylindrical decks) -------
#
# In cylindrical (r, z) mode the solver stores fields in Cartesian-named slots by
# an IMPLICIT convention that is a silent-wrong-result footgun if a user spells a
# raw lab/slot vector by hand: the grid triad is (i=r, j=z, out-of-plane=phi), so
# physically  B_r -> bx slot,  B_z (axial) -> by slot,  B_phi -> bz slot  (and the
# same for E and the velocity triad vx->vr, vy->vz, vz->vphi).
#
# The external-field sampler (src/physics/pic/external_field_sampler.cpp) consumes
# `uniform_b` as a LAB-AXIS vector and then applies a plane-dependent lab->slot map
# when it writes the six field slots:
#   plane "xy":  slot bx<-lab x,  by<-lab y,       bz<-lab z
#   plane "xz":  slot bx<-lab x,  by<-lab z,       bz<- -lab y
#
# So to make a physical `B_rzphi = [B_r, B_z, B_phi]` land in the (bx, by, bz)
# slots as (B_r, B_z, B_phi), we must hand the sampler the LAB vector that the
# plane map turns into those slots. Inverting the map above:
#   plane "xy":  uniform_b(lab) = [B_r, B_z,  B_phi]   (identity)
#   plane "xz":  uniform_b(lab) = [B_r, -B_phi, B_z]
#
# THIS FUNCTION IS THE ONE PLACE the physical (r,z,phi) ordering is converted to
# the lab-vector slot ordering for external uniform fields. Do not re-encode the
# permutation at any call site.
def _rzphi_to_uniform_lab(rzphi: Sequence[float], plane: str,
                          context: str) -> tuple[float, float, float]:
    br, bz, bphi = _triple(rzphi)
    if plane == "xz":
        return (br, -bphi, bz)
    # plane "xy" (default): the lab->slot map is the identity, so physical
    # (r, z, phi) already lines up with (bx, by, bz).
    return (br, bz, bphi)


def _parse_external_field(d: dict | None, geometry: str = "cartesian",
                          plane: str = "xy") -> Union[ExternalField, None]:
    if d is None:
        return None
    ev = _require(d, "evaluator", "external_field")
    ev_type = str(_require(ev, "type", "external_field.evaluator"))
    # Uniform external fields may use either terse or unit-explicit names.
    b = ev.get("B_T", ev.get("b_tesla", ev.get("B")))
    e = ev.get("E_V_per_m", ev.get("e_v_per_m", ev.get("E")))
    # Cylindrical decks may instead spell the uniform B by PHYSICAL axis,
    # `B_rzphi: [B_r, B_z, B_phi]`, which is translated to the lab/slot ordering
    # the sampler expects in exactly one place (_rzphi_to_uniform_lab). This is
    # the recommended spelling for cylindrical mode: it removes the need to know
    # the implicit "axial B goes in the second (by) slot" convention. The raw
    # `B_T` slot spelling still works (backward compatible); supplying both is an
    # error so a deck cannot silently disagree with itself.
    b_rzphi = ev.get("B_rzphi")
    if b_rzphi is not None:
        if geometry != "cylindrical":
            raise ValueError(
                "external_field.evaluator.B_rzphi (physical r,z,phi axes) is only "
                "valid for geometry 'cylindrical'; use B_T for cartesian decks")
        if b is not None:
            raise ValueError(
                "external_field.evaluator: give either B_rzphi (physical axes) or "
                "B_T (storage slots), not both")
        b = list(_rzphi_to_uniform_lab(b_rzphi, plane,
                                       "external_field.evaluator.B_rzphi"))
    # Dipole/gradient parameter names keep the SI unit in the key; shorter aliases
    # are accepted for normalized decks and for hand-written tests.
    moment = ev.get("moment_Am2", ev.get("moment_A_m2", ev.get("moment")))
    origin = ev.get("origin_xyz_m", ev.get("origin", [0.0, 0.0, 0.0]))
    b0 = ev.get("B0_T", ev.get("b0_tesla", ev.get("b0", [0.0, 0.0, 0.0])))
    grad = ev.get("grad_T_per_m", ev.get("gradient_T_per_m", ev.get("gradient")))
    return ExternalField(evaluator_type=ev_type,
                         conductors=list(ev.get("conductors", [])),
                         uniform_b=_vec3(b),
                         uniform_e=_vec3(e),
                         dipole_moment=None if moment is None else _vec3(moment),
                         dipole_origin=_vec3(origin),
                         gradient_b0=_vec3(b0),
                         gradient_matrix=_matrix3(grad),
                         gradient_origin=_vec3(origin))


def _parse_species(items: list[dict] | None) -> list[Species]:
    if not items:
        return []
    out: list[Species] = []
    for raw in items:
        name = str(_require(raw, "name", "species"))
        init_raw = raw.get("initial", {})
        region_raw = init_raw.get("region", {}) or {}
        def _opt(key):
            v = region_raw.get(key)
            return None if v is None else float(v)
        initial = SpeciesInitial(
            distribution=str(init_raw.get("distribution", "maxwellian_uniform")),
            density_per_m3=float(init_raw.get("density_per_m3", 1.0e18)),
            temperature_eV=float(init_raw.get("temperature_eV", 1.0)),
            drift_v=_vec3(init_raw.get("drift_v")),
            region_x_min_m=_opt("x_min_m"),
            region_x_max_m=_opt("x_max_m"),
            region_y_min_m=_opt("y_min_m"),
            region_y_max_m=_opt("y_max_m"),
        )
        out.append(Species(
            name=name,
            charge_C=float(_require(raw, "charge_C", f"species {name!r}")),
            mass_kg=float(_require(raw, "mass_kg", f"species {name!r}")),
            n_particles=int(_require(raw, "n_particles", f"species {name!r}")),
            initial=initial,
        ))
    return out


def _parse_time(d: dict | None) -> Time:
    if d is None:
        return Time()
    dt_raw = d.get("dt_s", "auto")
    dt_s: Union[float, str] = dt_raw if isinstance(dt_raw, str) else float(dt_raw)
    return Time(dt_s=dt_s, steps=int(d.get("steps", 100)))


def _parse_diagnostics(d: dict | None) -> Diagnostics:
    if d is None:
        return Diagnostics()
    return Diagnostics(
        output_path=str(d.get("output_path", "out.npz")),
        cadence=int(d.get("cadence", 0)),
        fields=list(d.get("fields", ["bz"])),
        per_species=bool(d.get("per_species", True)),
    )


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
        particle=_parse_side_map(d.get("particle"), "periodic", "boundary.particle"),
        field=_parse_side_map(d.get("field"), "periodic", "boundary.field"),
    )


# --- fields.initial physical-component -> storage-slot translation ------------
#
# THIS IS THE ONE PLACE the cylindrical physical seed-component names are mapped
# to Cartesian storage slots. Using the same grid triad as the external-field
# translation above (i=r, j=z, out-of-plane=phi):
#   E_r -> ex,  E_z (axial) -> ey,  E_phi (azimuthal) -> ez   (and B analogously)
# In cylindrical mode the names Er/Ez/Ephi/Br/Bz/Bphi are REINTERPRETED as these
# physical axes, so `component: Ez` means the AXIAL field and resolves to the `ey`
# slot (NOT the literal `ez` storage slot). To seed the azimuthal field, spell it
# `Ephi`. Raw slot names (ex/ey/ez/bx/by/bz) are still accepted unchanged for
# backward compatibility, but the physical spelling is recommended for clarity.
# In CARTESIAN mode no reinterpretation happens: a slot name is taken literally.
_CYL_PHYSICAL_COMPONENT = {
    "er": "ex", "ez": "ey", "ephi": "ez",
    "br": "bx", "bz": "by", "bphi": "bz",
}
# Raw storage-slot names, accepted verbatim in either geometry.
_FIELD_SLOTS = ("ex", "ey", "ez", "bx", "by", "bz")


def _resolve_seed_component(name: str, geometry: str) -> str:
    """Map a deck seed-component name to a storage slot (ex/ey/.../bz).

    Cylindrical: physical Er/Ez/Ephi/Br/Bz/Bphi -> slot via the triad map above;
    a raw slot name passes through. Cartesian: slot names pass through verbatim.
    """
    if geometry == "cylindrical":
        key = name.strip().lower()
        if key in _CYL_PHYSICAL_COMPONENT:
            return _CYL_PHYSICAL_COMPONENT[key]
        if key in _FIELD_SLOTS:
            return key  # explicit slot name, backward-compatible passthrough
        raise ValueError(
            f"fields.initial.component {name!r} is not a recognized cylindrical "
            f"physical component (Er/Ez/Ephi/Br/Bz/Bphi) or storage slot "
            f"({list(_FIELD_SLOTS)})")
    # cartesian: return the name VERBATIM (original case preserved, as before the
    # physical-component refactor). The cli seeder lowercases at use-time, so case
    # here is purely the parsed-value contract that existing decks/tests rely on.
    return name.strip()


def _parse_fields(d: dict | None, geometry: str = "cartesian") -> Fields:
    if d is None:
        return Fields()
    init = d.get("initial")
    if init is None:
        return Fields()
    mode_raw = init.get("mode", [1, 0])
    if isinstance(mode_raw, int):
        mode = (int(mode_raw), 0)
    else:
        mode = (int(mode_raw[0]), int(mode_raw[1]) if len(mode_raw) > 1 else 0)
    raw_component = str(init.get("component", "Ex"))
    return Fields(initial=FieldsInitial(
        # Resolve the physical/slot name to a storage slot here, once, so every
        # downstream consumer (cli._seed_fields) sees a plain slot name.
        type=str(_require(init, "type", "fields.initial")),
        component=_resolve_seed_component(raw_component, geometry),
        amplitude=float(init.get("amplitude", 1.0e-4)),
        mode=mode,
    ))


def parse(data: dict) -> PicDeck:
    # Geometry/plane are needed by the external-field and fields.initial parsers so
    # they can translate the physical-axis spellings to storage slots, so resolve
    # them before constructing the deck.
    geometry = str(data.get("geometry", "cartesian"))
    plane = str(data.get("plane", "xy"))
    deck = PicDeck(
        domain=_parse_domain(_require(data, "domain", "deck")),
        numerics=_parse_numerics(data.get("numerics")),
        normalization=_parse_normalization(data.get("normalization")),
        external_field=_parse_external_field(data.get("external_field"),
                                             geometry, plane),
        species=_parse_species(data.get("species")),
        time=_parse_time(data.get("time")),
        diagnostics=_parse_diagnostics(data.get("diagnostics")),
        boundary=_parse_boundary(data.get("boundary")),
        fields=_parse_fields(data.get("fields"), geometry),
        units=str(data.get("units", "SI")),
        plane=plane,
        geometry=geometry,
        raw=data,
    )
    deck.validate()
    return deck


def load(path: Union[str, Path]) -> PicDeck:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return parse(data)
