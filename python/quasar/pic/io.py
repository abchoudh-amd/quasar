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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

import yaml

from .._deck import require as _require, triple as _triple
from .._deck import validate_evaluator_type as _validate_evaluator_type


# Sanity ceilings on deck-supplied sizes that flow into device allocations.
# These guard against typos / hostile decks requesting absurd buffers, not
# against legitimate large runs (raise them if a real workload needs more).
MAX_GRID_DIM = 1 << 16        # 65536 cells per axis
MAX_GRID_CELLS = 1 << 30      # ~1.07e9 cells total
MAX_PARTICLES = 1 << 31       # ~2.1e9 particles per species


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
    raw: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.units not in ("SI", "normalized"):
            raise ValueError("units must be 'SI' or 'normalized'")
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
        if self.domain.lx_m <= 0 or self.domain.ly_m <= 0:
            raise ValueError("domain.lx_m and domain.ly_m must be positive")
        if self.numerics.fdtd_order not in (2, 4):
            raise ValueError("numerics.fdtd_order must be 2 or 4")
        if self.numerics.shape not in ("cic", "tsc"):
            raise ValueError("numerics.shape must be 'cic' or 'tsc'")
        allowed_filters = {"binomial", "compensated_binomial"}
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
        if self.external_field is not None:
            ev = self.external_field.evaluator_type
            _validate_evaluator_type(ev, "external_field.evaluator.type")
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
        # A deck must drive *something*: particles, an external field, or an
        # initial field seed. Field-only decks (e.g. EM-wave propagation, coil
        # confinement) legitimately have no species.
        if (not self.species and self.external_field is None
                and self.fields.initial is None):
            raise ValueError(
                "deck must define at least one of: species, external_field, "
                "or fields.initial")
        for sp in self.species:
            if sp.mass_kg <= 0:
                raise ValueError(f"species {sp.name!r}: mass_kg must be positive")
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
            if sp.initial.temperature_eV < 0:
                raise ValueError(f"species {sp.name!r}: temperature_eV must be >= 0")
        if isinstance(self.time.dt_s, str) and self.time.dt_s != "auto":
            raise ValueError("time.dt_s must be a float or the string 'auto'")
        if self.time.steps <= 0:
            raise ValueError("time.steps must be positive")
        allowed_pbc = {"periodic", "specular", "absorbing"}
        for i, bc in enumerate(self.boundary.particle):
            if bc not in allowed_pbc:
                raise ValueError(
                    f"boundary.particle[{i}] = {bc!r} must be one of {sorted(allowed_pbc)}")
        allowed_fbc = {"periodic", "pec", "outflow"}
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


def _parse_external_field(d: dict | None) -> Union[ExternalField, None]:
    if d is None:
        return None
    ev = _require(d, "evaluator", "external_field")
    ev_type = str(_require(ev, "type", "external_field.evaluator"))
    # Uniform external fields may use either terse or unit-explicit names.
    b = ev.get("B_T", ev.get("b_tesla", ev.get("B")))
    e = ev.get("E_V_per_m", ev.get("e_v_per_m", ev.get("E")))
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


def _parse_fields(d: dict | None) -> Fields:
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
    return Fields(initial=FieldsInitial(
        type=str(_require(init, "type", "fields.initial")),
        component=str(init.get("component", "Ex")),
        amplitude=float(init.get("amplitude", 1.0e-4)),
        mode=mode,
    ))


def parse(data: dict) -> PicDeck:
    deck = PicDeck(
        domain=_parse_domain(_require(data, "domain", "deck")),
        numerics=_parse_numerics(data.get("numerics")),
        normalization=_parse_normalization(data.get("normalization")),
        external_field=_parse_external_field(data.get("external_field")),
        species=_parse_species(data.get("species")),
        time=_parse_time(data.get("time")),
        diagnostics=_parse_diagnostics(data.get("diagnostics")),
        boundary=_parse_boundary(data.get("boundary")),
        fields=_parse_fields(data.get("fields")),
        units=str(data.get("units", "SI")),
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
