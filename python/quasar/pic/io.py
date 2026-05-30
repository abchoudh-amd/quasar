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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, Union

import yaml


# Sanity ceilings on deck-supplied sizes that flow into device allocations.
# These guard against typos / hostile decks requesting absurd buffers, not
# against legitimate large runs (raise them if a real workload needs more).
MAX_GRID_DIM = 1 << 16        # 65536 cells per axis
MAX_GRID_CELLS = 1 << 30      # ~1.07e9 cells total
MAX_PARTICLES = 1 << 31       # ~2.1e9 particles per species


def _require(d: dict, key: str, context: str) -> Any:
    if key not in d:
        raise ValueError(f"{context}: missing required field {key!r}")
    return d[key]


def _vec3(xyz: Sequence[float] | None,
          default: tuple[float, float, float] = (0.0, 0.0, 0.0),
          ) -> tuple[float, float, float]:
    if xyz is None:
        return default
    if len(xyz) != 3:
        raise ValueError(f"expected 3-element triple, got {xyz!r}")
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


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
    """Per-side particle boundary kinds. Order: [x_min, x_max, y_min, y_max].

    Each entry is one of ``periodic`` (no-op), ``specular`` (reflect), or
    ``absorbing`` (mark out-of-domain particles as dead).
    """
    particle: tuple[str, str, str, str] = (
        "periodic", "periodic", "periodic", "periodic")


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
        if self.external_field is not None:
            if self.external_field.evaluator_type != "biot_savart":
                raise ValueError(
                    "external_field.evaluator.type must be 'biot_savart' "
                    "(only Biot-Savart is bound to Python today)"
                )
            if not self.external_field.conductors:
                raise ValueError("external_field.evaluator.conductors must be non-empty")
        if not self.species:
            raise ValueError("deck.species must be non-empty")
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
        allowed_bc = {"periodic", "specular", "absorbing"}
        for i, bc in enumerate(self.boundary.particle):
            if bc not in allowed_bc:
                raise ValueError(
                    f"boundary.particle[{i}] = {bc!r} must be one of {sorted(allowed_bc)}")


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
    return ExternalField(evaluator_type=ev_type,
                         conductors=list(ev.get("conductors", [])))


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


def _parse_boundary(d: dict | None) -> BoundaryConfig:
    if d is None:
        return BoundaryConfig()
    p = d.get("particle", "periodic")
    if isinstance(p, str):
        particle = (p, p, p, p)
    elif isinstance(p, (list, tuple)) and len(p) == 4:
        particle = (str(p[0]), str(p[1]), str(p[2]), str(p[3]))
    else:
        raise ValueError("boundary.particle must be a string or 4-element list")
    return BoundaryConfig(particle=particle)


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


def build_conductor_system(conductors: list[dict]):
    """Build a ConductorSystem from a list of conductor dicts.

    Thin re-export of ``quasar.coil.io.build_conductor_system`` so the PIC
    external-field loader shares the conductor schema with the coil workflow.
    """
    from ..coil.io import build_conductor_system as _build
    return _build(conductors)
