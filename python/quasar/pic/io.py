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
          density_per_m3, temperature_eV, drift_v?, velocity_perturbation?}
    time: {dt_s, steps, t_end_s?}
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

from .. import _core
from .._deck import as_boolean as _as_boolean
from .._deck import as_integer as _as_integer
from .._deck import load_yaml as _load_yaml
from .._deck import require as _require, triple as _triple
from .._deck import unique_alias as _unique_alias
from .._deck import validate_evaluator_type as _validate_evaluator_type
from .._field_grid import load_file_grid_npz
from .._deck import (
    flat_evaluator_params as _flat_evaluator_params,
    require_finite as _require_finite,
    require_positive_finite as _require_positive_finite,
    require_nonnegative_finite as _require_nonnegative_finite,
    require_vec_finite as _require_vec_finite,
    parse_side_map as _parse_side_map,
)


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
    file_path: str | None = None
    file_grid_params: dict[str, list[float]] = field(default_factory=dict)
    # Registry-provided evaluators use this generic configure() payload. Built-in
    # evaluator types retain their unit-aware named fields above so the deck can
    # validate and translate their physical parameters explicitly.
    params: dict[str, list[float]] = field(default_factory=dict)

    def evaluator_params(self) -> dict[str, list[float]]:
        """Deck parameters as the ``name -> flat float list`` map the C++
        ``IFieldEvaluator.configure`` seam consumes (Vec3 = 3 elements, Mat3x3 = 9
        row-major). Only keys belonging to the selected evaluator are emitted;
        configure() deliberately rejects unknown keys.
        """
        if self.evaluator_type == "file_grid":
            return dict(self.file_grid_params)
        if self.evaluator_type == "biot_savart":
            return {}
        if self.evaluator_type == "uniform":
            return {"b0": list(self.uniform_b), "e0": list(self.uniform_e)}
        if self.evaluator_type == "dipole":
            result = {"origin": list(self.dipole_origin)}
            if self.dipole_moment is not None:
                result["moment"] = list(self.dipole_moment)
            return result
        if self.evaluator_type == "gradient":
            result = {
                "b0": list(self.gradient_b0),
                "origin": list(self.gradient_origin),
            }
            if self.gradient_matrix is not None:
                result["grad"] = [
                    value for row in self.gradient_matrix for value in row]
            return result
        return _flat_evaluator_params(
            self.params, "external_field.evaluator.params")


@dataclass
class VelocityPerturbation:
    """Resolved sinusoidal perturbation added to a species' drift velocity.

    ``amplitude_v`` uses the deck's velocity units.  ``mode=(mx,my)`` counts
    full wavelengths across the simulation domain and ``phase_rad`` is in
    radians.  A common perturbation on two counter-streaming beams seeds a
    deterministic current mode without introducing an initial charge-density
    error.
    """
    amplitude_v: tuple[float, float, float]
    mode: tuple[int, int] = (1, 0)
    phase_rad: float = 0.0


@dataclass
class SpeciesInitial:
    distribution: str = "maxwellian_uniform"
    density_per_m3: float = 1.0e18
    temperature_eV: float = 1.0
    drift_v: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_perturbation: Union[VelocityPerturbation, None] = None
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
    # Optional physical/internal end time (SI seconds for SI decks, identity for
    # normalized decks). ``steps`` remains a safety cap; the CLI clips the final
    # step so a reachable t_end_s is hit exactly.
    t_end_s: Union[float, None] = None


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

    * ``seed_perturbation``: a divergence-free single-mode profile in one
      transverse field component (``component``, ``amplitude``, ``mode``).
      Longitudinal ``Ex``/``Bx`` seeds are rejected because a lone sinusoid
      would violate Gauss's law or ``div(B)=0``. Amplitude is in V/m or tesla
      for an SI deck, according to the selected E/B component, and is already
      in solver units for a normalized deck.
    * ``seed_em_wave``: a Cartesian plane wave propagating in ``+x``
      (``component`` Ez or Ey with its magnetic partner, ``mode=(mx, 0)``).
    * ``seed_tm_cavity``: a Cartesian rectangular-PEC ``TM_mn`` eigenmode.
      ``component`` must be Ez, ``mode=(m, n)`` has two positive wall-mode
      indices, and the seeder initializes Ez at integer time together with Bx/By
      at the preceding half time using the selected Yee operator's discrete
      dispersion relation.
    """
    type: str
    component: str = "Ey"
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
    # plane with grid axes (x,z). Cartesian xz uses the right-handed third basis
    # -lab-y; cylindrical xz instead stores the physical +phi component, which is
    # +lab-y on the positive-x meridian (see the single slot map below).
    plane: str = "xy"
    # Coordinate system of the 2D grid. "cartesian" (default) treats axes as
    # (x, y) in the chosen plane; "cylindrical" treats them as axisymmetric r-z
    # with i=r (from nx / lx_m / origin_x_m) and j=z. The C++ solver applies the
    # on-axis closure for the inner-radius (x_lo) boundary.
    geometry: str = "cartesian"
    # Explicit fixed background charge model. When true, the C++ solver computes
    # a uniform initial background that cancels total particle charge.
    neutralizing_background: bool = False
    raw: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.units not in ("SI", "normalized"):
            raise ValueError("units must be 'SI' or 'normalized'")
        if self.plane not in ("xy", "xz"):
            raise ValueError("plane must be 'xy' or 'xz'")
        if self.geometry not in ("cartesian", "cylindrical"):
            raise ValueError(
                "geometry must be 'cartesian' or 'cylindrical'")
        _as_boolean(self.neutralizing_background, "neutralizing_background")
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
            init = self.fields.initial
            _require_finite(init.amplitude, "fields.initial.amplitude")
            if init.type not in (
                    "seed_perturbation", "seed_em_wave", "seed_tm_cavity"):
                raise ValueError(
                    "fields.initial.type must be 'seed_perturbation', "
                    "'seed_em_wave', or 'seed_tm_cavity'")
            component = init.component.lower()
            if component not in FIELD_COMPONENTS:
                raise ValueError(
                    f"fields.initial.component {init.component!r} must be one of "
                    f"{list(FIELD_COMPONENTS)}")
            if len(init.mode) != 2:
                raise ValueError("fields.initial.mode must contain exactly two integers")
            mx = _as_integer(init.mode[0], "fields.initial.mode[0]")
            my = _as_integer(init.mode[1], "fields.initial.mode[1]")
            if init.type == "seed_tm_cavity":
                if self.geometry != "cartesian":
                    raise ValueError(
                        "seed_tm_cavity is defined only for Cartesian rectangular "
                        "cavities")
                if component != "ez":
                    raise ValueError(
                        "seed_tm_cavity component must be Ez (the out-of-plane "
                        "electric field)")
                if mx < 1 or my < 1:
                    raise ValueError(
                        "seed_tm_cavity mode=(m,n) requires m >= 1 and n >= 1")
                # A cell-centred Dirichlet lattice has exactly nx (ny) distinct
                # sine eigenvectors, including its alternating highest mode.
                if mx > self.domain.nx or my > self.domain.ny:
                    raise ValueError(
                        "seed_tm_cavity mode exceeds the rectangular Yee "
                        "Dirichlet spectrum")
                if any(name != "pec" for name in self.boundary.field):
                    raise ValueError(
                        "seed_tm_cavity requires PEC field boundaries on all "
                        "four sides")
            else:
                if mx < 1 or my != 0:
                    raise ValueError(
                        "fields.initial.mode requires mx >= 1 and my == 0; only "
                        "x/r-directed seeds are implemented")
                if self.geometry == "cartesian":
                    if (init.type == "seed_em_wave"
                            and 2 * mx >= self.domain.nx):
                        # At equality the +k and -k samples alias, so the
                        # requested direction is not representable even though
                        # a checkerboard eigenvector exists.
                        raise ValueError(
                            "seed_em_wave mode must lie strictly below the x "
                            "Nyquist mode to represent a directional "
                            "travelling wave")
                    if 2 * mx > self.domain.nx:
                        raise ValueError(
                            "fields.initial.mode[0] exceeds the resolved x "
                            "Nyquist mode")
                if (init.type == "seed_em_wave"
                        and component not in ("ey", "ez")):
                    raise ValueError(
                        "seed_em_wave component must be Ey or Ez (storage slots "
                        "ey/ez)")
            if (self.geometry == "cartesian"
                    and init.type == "seed_perturbation"
                    and component in ("ex", "bx")):
                raise ValueError(
                    "cartesian seed_perturbation cannot seed longitudinal Ex "
                    "or Bx: a single x-varying component would violate Gauss's "
                    "law or div(B)=0; use a transverse component")
            if self.geometry == "cylindrical":
                if init.type == "seed_em_wave":
                    raise ValueError(
                        "seed_em_wave is invalid for cylindrical geometry: a "
                        "radially translating Cartesian plane wave is not a "
                        "regular axisymmetric mode; use an axial "
                        "seed_perturbation Bessel cavity mode")
                if component != "ey":
                    raise ValueError(
                        "cylindrical seed_perturbation currently supports only "
                        "physical axial Ez (storage component Ey)")
                if self.domain.origin_x_m != 0.0:
                    raise ValueError(
                        "cylindrical seed_perturbation requires an on-axis "
                        "domain (origin_x_m=0); annular eigenmodes require a "
                        "J/Y Bessel combination")
                if self.boundary.field[1] != "pec":
                    raise ValueError(
                        "cylindrical seed_perturbation is a J0 cavity mode and "
                        "requires a PEC outer-radius (x_hi) field boundary")
                # j_{0,mx}/R is the largest local radial wavenumber of this
                # profile.  Since j_{0,n} lies strictly between
                # (n-1/4) pi and n pi, mx <= nx is exactly the set of these
                # modes below the radial mesh Nyquist wavenumber pi/dr.
                if mx > self.domain.nx:
                    raise ValueError(
                        "cylindrical fields.initial.mode[0] exceeds the "
                        "resolved radial Nyquist spectrum")
        self._validate_time()
        self._validate_diagnostics()
        # Radial topology has more specific physical constraints than the generic
        # periodic-pair checks below. Diagnose those first so a cylindrical deck
        # never reports a Cartesian topology symptom in place of the actual
        # negative/periodic radius error.
        if self.geometry == "cylindrical":
            self._validate_cylindrical()
        self._validate_boundary()

    def _validate_cylindrical(self) -> None:
        # Negative radius has no cylindrical interpretation.  r_min=0 selects
        # the regular parity axis closure; r_min>0 is an annulus and retains the
        # explicitly configured x_lo field/particle boundary.
        if self.domain.origin_x_m < 0.0:
            raise ValueError(
                "geometry 'cylindrical': domain.origin_x_m must be >= 0")
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
        if self.domain.origin_x_m > 0.0:
            if self.boundary.field[0] == "periodic":
                raise ValueError(
                    "geometry 'cylindrical' annulus: boundary.field x_lo "
                    "(inner radius) must not be 'periodic'")
            if self.boundary.particle[0] == "periodic":
                raise ValueError(
                    "geometry 'cylindrical' annulus: boundary.particle x_lo "
                    "(inner radius) must not be 'periodic'")

    def _validate_domain(self) -> None:
        _as_integer(self.domain.nx, "domain.nx")
        _as_integer(self.domain.ny, "domain.ny")
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
        upper_x = self.domain.origin_x_m + self.domain.lx_m
        upper_y = self.domain.origin_y_m + self.domain.ly_m
        _require_finite(upper_x, "domain upper x bound")
        _require_finite(upper_y, "domain upper y bound")
        dx = self.domain.lx_m / self.domain.nx
        dy = self.domain.ly_m / self.domain.ny
        _require_positive_finite(dx, "domain x spacing")
        _require_positive_finite(dy, "domain y spacing")
        if (self.domain.origin_x_m + 0.5 * dx == self.domain.origin_x_m
                or (self.domain.origin_x_m
                    + (self.domain.nx - 0.5) * dx) == upper_x
                or self.domain.origin_y_m + 0.5 * dy == self.domain.origin_y_m
                or (self.domain.origin_y_m
                    + (self.domain.ny - 0.5) * dy) == upper_y):
            raise ValueError(
                "domain cell coordinates collapse in floating-point precision")
        _require_positive_finite(
            self.normalization.reference_density_per_m3,
            "normalization.reference_density_per_m3")

    def _validate_numerics(self) -> None:
        _as_integer(self.numerics.fdtd_order, "numerics.fdtd_order")
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
            passes = _as_integer(
                _unique_alias(
                    spec, ("n_passes", "passes"),
                    f"numerics.current_filter[{i}] pass count", 1),
                f"numerics.current_filter[{i}].passes")
            if passes < 1:
                raise ValueError(
                    f"numerics.current_filter[{i}] passes must be >= 1")

    def _validate_external_field(self) -> None:
        if self.external_field is None:
            return
        ev = self.external_field.evaluator_type
        if self.units == "normalized" and ev in ("biot_savart", "dipole"):
            raise ValueError(
                f"external_field evaluator {ev!r} is SI-defined and has no "
                "unambiguous current/magnetic-moment scale for normalized decks; "
                "use units: SI or provide a pre-normalized uniform/gradient field")
        _validate_evaluator_type(ev, "external_field.evaluator.type")
        builtins = {
            "biot_savart", "uniform", "dipole", "gradient", "file_grid"}
        if ev in builtins:
            if self.external_field.params:
                raise ValueError(
                    "external_field.evaluator.params is reserved for "
                    "registered plugin evaluators; use the built-in "
                    f"{ev!r} parameter fields")
        else:
            _flat_evaluator_params(
                self.external_field.params,
                "external_field.evaluator.params")
        _require_vec_finite(
            self.external_field.uniform_b, "external_field.evaluator.B")
        _require_vec_finite(
            self.external_field.uniform_e, "external_field.evaluator.E")
        if self.geometry == "cylindrical" and ev == "uniform":
            br, _bz, bphi = _uniform_lab_to_rzphi(
                self.external_field.uniform_b, self.plane)
            er, _ez, ephi = _uniform_lab_to_rzphi(
                self.external_field.uniform_e, self.plane)
            # `uniform` is a constant vector in the Cartesian lab basis.  A
            # nonzero cylindrical Br/E_r or Bphi/Ephi direction must rotate with
            # azimuth and therefore cannot be represented by that evaluator,
            # even on an annulus where a genuine axisymmetric toroidal profile
            # could be regular.  Reinterpreting only its meridional slice would
            # silently turn a nonaxisymmetric 3-D field into an m=0 one.
            if br != 0.0 or bphi != 0.0:
                raise ValueError(
                    "a lab-uniform cylindrical magnetic evaluator may have only "
                    "axial Bz; Br/Bphi require an axisymmetric nonuniform "
                    "evaluator")
            if er != 0.0 or ephi != 0.0:
                raise ValueError(
                    "a lab-uniform cylindrical electric evaluator may have only "
                    "axial Ez; Er/Ephi require an axisymmetric nonuniform "
                    "evaluator")
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
        if ev == "gradient" and self.external_field.gradient_matrix is not None:
            matrix = self.external_field.gradient_matrix
            trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
            scale = max(abs(matrix[0][0]), abs(matrix[1][1]), abs(matrix[2][2]))
            if abs(trace) > 64.0 * math.ulp(1.0) * scale:
                raise ValueError(
                    "external_field.evaluator.grad_T_per_m must be trace-free "
                    "(Maxwell div(B)=0)")
        if ev == "file_grid" and (
                not isinstance(self.external_field.file_path, str)
                or not self.external_field.file_path.strip()):
            raise ValueError(
                "external_field.evaluator.path is required for type 'file_grid'")

    def _validate_species(self) -> None:
        names: set[str] = set()
        archive_keys: dict[str, tuple[str, str]] = {}
        for sp in self.species:
            if (not isinstance(sp.name, str) or not sp.name
                    or sp.name != sp.name.strip()):
                raise ValueError(
                    "species names must be nonempty strings without leading "
                    "or trailing whitespace")
            if sp.name in names:
                raise ValueError(f"duplicate species name {sp.name!r}")
            names.add(sp.name)
            # The stable NPZ schema is species_<name>_<field>. Unique names alone
            # are not enough to make that encoding injective: ("a", "vx") and
            # ("a_v", "x") both spell species_a_vx. Reject only combinations
            # that actually collide, preserving established descriptive names.
            for field_name in ("x", "y", "vx", "vy", "vz", "weight", "alive"):
                key = f"species_{sp.name}_{field_name}"
                previous = archive_keys.get(key)
                if previous is not None:
                    raise ValueError(
                        "species names are ambiguous in the output schema: "
                        f"{previous[0]!r}/{previous[1]!r} and "
                        f"{sp.name!r}/{field_name!r} both map to {key!r}")
                archive_keys[key] = (sp.name, field_name)
            _as_integer(sp.n_particles, f"species {sp.name!r}: n_particles")
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
                domain_x_hi = self.domain.origin_x_m + self.domain.lx_m
                domain_y_hi = self.domain.origin_y_m + self.domain.ly_m
                if (sp.initial.region_x_min_m < self.domain.origin_x_m
                        or sp.initial.region_x_max_m > domain_x_hi
                        or sp.initial.region_y_min_m < self.domain.origin_y_m
                        or sp.initial.region_y_max_m > domain_y_hi):
                    raise ValueError(
                        f"species {sp.name!r}: initial region must lie within "
                        "the simulation domain")
            _require_nonnegative_finite(
                sp.initial.density_per_m3,
                f"species {sp.name!r}: density_per_m3")
            _require_nonnegative_finite(
                sp.initial.temperature_eV,
                f"species {sp.name!r}: temperature_eV")
            _require_vec_finite(sp.initial.drift_v, f"species {sp.name!r}: drift_v")
            perturbation = sp.initial.velocity_perturbation
            if perturbation is not None:
                _require_vec_finite(
                    perturbation.amplitude_v,
                    f"species {sp.name!r}: velocity_perturbation.amplitude_v")
                if not any(value != 0.0 for value in perturbation.amplitude_v):
                    raise ValueError(
                        f"species {sp.name!r}: velocity_perturbation amplitude "
                        "must be nonzero")
                if len(perturbation.mode) != 2:
                    raise ValueError(
                        f"species {sp.name!r}: velocity_perturbation.mode must "
                        "contain exactly two integers")
                mx = _as_integer(
                    perturbation.mode[0],
                    f"species {sp.name!r}: velocity_perturbation.mode[0]")
                my = _as_integer(
                    perturbation.mode[1],
                    f"species {sp.name!r}: velocity_perturbation.mode[1]")
                if mx == 0 and my == 0:
                    raise ValueError(
                        f"species {sp.name!r}: velocity_perturbation.mode must "
                        "contain a nonzero wave number")
                _require_finite(
                    perturbation.phase_rad,
                    f"species {sp.name!r}: velocity_perturbation.phase_rad")
                for mode, cells, axis in (
                        (mx, self.domain.nx, "x/r"),
                        (my, self.domain.ny, "y/z")):
                    # The centered Maxwell/deposit mesh has no unique propagating
                    # representation at or above Nyquist.  In particular, the
                    # even-grid Nyquist sine can collapse or alias according to
                    # phase and particle placement.  Require a strictly resolved
                    # wavelength instead of accepting a deck whose nominal mode
                    # is not the mode the solver evolves.
                    if 2 * abs(mode) >= cells:
                        raise ValueError(
                            f"species {sp.name!r}: velocity_perturbation mode "
                            f"{mode} on {axis} must lie strictly below the "
                            f"{cells}-cell Nyquist limit")

                if (self.geometry == "cylindrical"
                        and self.domain.origin_x_m == 0.0):
                    # At r=0, vr and vphi are odd while axial vz is even.  The
                    # shared scalar sine profile can meet those parities only in
                    # the cases below.  Test the phase modulo 2*pi so equivalent
                    # finite spellings receive identical validation.
                    phase = math.remainder(
                        perturbation.phase_rad, 2.0 * math.pi)
                    parity_tol = 256.0 * math.ulp(1.0)
                    transverse = (
                        perturbation.amplitude_v[0] != 0.0
                        or perturbation.amplitude_v[2] != 0.0)
                    axial = perturbation.amplitude_v[1] != 0.0
                    if transverse and (
                            mx == 0 or my != 0
                            or abs(math.sin(phase)) > parity_tol):
                        raise ValueError(
                            f"species {sp.name!r}: cylindrical vr/vphi velocity "
                            "perturbations on an r=0 domain require a nonzero "
                            "radial mode, zero axial mode, and phase = 0 mod pi "
                            "so the vector vanishes on the axis")
                    if axial and mx != 0 and (
                            my != 0
                            or abs(math.cos(phase)) > parity_tol):
                        raise ValueError(
                            f"species {sp.name!r}: a radial variation of axial "
                            "velocity on an r=0 domain requires zero axial mode "
                            "and phase = pi/2 mod pi so its radial derivative "
                            "vanishes on the axis")

    def _validate_time(self) -> None:
        _as_integer(self.time.steps, "time.steps")
        if isinstance(self.time.dt_s, str) and self.time.dt_s != "auto":
            raise ValueError("time.dt_s must be a float or the string 'auto'")
        if not isinstance(self.time.dt_s, str):
            _require_positive_finite(self.time.dt_s, "time.dt_s")
        if self.time.steps <= 0:
            raise ValueError("time.steps must be positive")
        if self.time.t_end_s is not None:
            _require_positive_finite(self.time.t_end_s, "time.t_end_s")

    def _validate_diagnostics(self) -> None:
        _as_integer(self.diagnostics.cadence, "diagnostics.cadence")
        _as_boolean(self.diagnostics.per_species, "diagnostics.per_species")
        if not self.diagnostics.output_path.strip():
            raise ValueError("diagnostics.output_path must not be empty")
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
        # The on-axis 'axis' condition only applies to the inner-radius side
        # (x_lo, index 0); the C++ BC no-ops on any other side, so selecting it
        # elsewhere is silent misuse. Reject it rather than letting it do nothing.
        # (x_lo is auto-wired to 'axis' for cylindrical decks; setting it there
        # explicitly is still allowed.)
        for i, bc in enumerate(self.boundary.particle):
            if bc == "axis" and not (
                    self.geometry == "cylindrical"
                    and self.domain.origin_x_m == 0.0 and i == 0):
                raise ValueError(
                    f"boundary.particle[{i}] = 'axis' is only valid at "
                    "cylindrical r=0 on x_lo")
        for i, bc in enumerate(self.boundary.field):
            if bc == "axis" and not (
                    self.geometry == "cylindrical"
                    and self.domain.origin_x_m == 0.0 and i == 0):
                raise ValueError(
                    f"boundary.field[{i}] = 'axis' is only valid at "
                    "cylindrical r=0 on x_lo")

        if self.geometry == "cylindrical" and self.domain.origin_x_m == 0.0:
            for name, sides in (("field", self.boundary.field),
                                ("particle", self.boundary.particle)):
                if sides[0] not in ("periodic", "axis"):
                    raise ValueError(
                        f"boundary.{name}[0] at cylindrical r=0 must be "
                        "'axis' (the default 'periodic' placeholder is "
                        "auto-replaced)")

        # Treat the default cylindrical x_lo placeholder exactly as the C++
        # solver does: at r=0 it becomes the non-periodic axis closure before
        # topology is checked. Every other periodic side must be paired, and
        # field/particle periodicity must agree axis by axis so deposited charge
        # and Maxwell divergence live on the same topological domain.
        field = list(self.boundary.field)
        particle = list(self.boundary.particle)
        if self.geometry == "cylindrical" and self.domain.origin_x_m == 0.0:
            field[0] = "axis"
            particle[0] = "axis"
        for name, sides in (("field", field), ("particle", particle)):
            for lo, hi, axis in ((0, 1, "x/r"), (2, 3, "y/z")):
                if ((sides[lo] == "periodic")
                        != (sides[hi] == "periodic")):
                    raise ValueError(
                        f"boundary.{name} periodic sides on the {axis} axis "
                        "must be specified as a pair")
        for lo, axis in ((0, "x/r"), (2, "y/z")):
            if ((field[lo] == "periodic")
                    != (particle[lo] == "periodic")):
                raise ValueError(
                    "boundary field and particle periodicity must match on "
                    f"the {axis} axis")


def _parse_domain(d: dict) -> Domain:
    return Domain(
        nx=_as_integer(_require(d, "nx", "domain"), "domain.nx"),
        ny=_as_integer(_require(d, "ny", "domain"), "domain.ny"),
        lx_m=float(_require(d, "lx_m", "domain")),
        ly_m=float(_require(d, "ly_m", "domain")),
        origin_x_m=float(d.get("origin_x_m", 0.0)),
        origin_y_m=float(d.get("origin_y_m", 0.0)),
    )


def _parse_numerics(d: dict | None) -> Numerics:
    if d is None:
        return Numerics()
    return Numerics(
        fdtd_order=_as_integer(d.get("fdtd_order", 2), "numerics.fdtd_order"),
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
# `uniform_b` as a LAB-AXIS vector and then applies a plane- and geometry-
# dependent lab->slot map when it writes the six field slots. For cylindrical
# storage specifically:
#   plane "xy":  slot bx<-lab x,  by<-lab y,       bz<- -lab z
#   plane "xz":  slot bx<-lab x,  by<-lab z,       bz<- +lab y
# Cartesian maps remain ordinary xy identity and the right-handed xz mapping
# (x,z,-y), respectively.
#
# So to make a physical `B_rzphi = [B_r, B_z, B_phi]` land in the (bx, by, bz)
# slots as (B_r, B_z, B_phi), we must hand the sampler the LAB vector that the
# cylindrical plane map turns into those slots. Inverting the map above:
#   plane "xy":  uniform_b(lab) = [B_r, B_z, -B_phi]
#   plane "xz":  uniform_b(lab) = [B_r, B_phi, B_z]
#
# THIS FUNCTION IS THE ONE PLACE the physical (r,z,phi) ordering is converted to
# the lab-vector slot ordering for external uniform fields. Do not re-encode the
# permutation at any call site.
def _rzphi_to_uniform_lab(rzphi: Sequence[float], plane: str,
                          context: str) -> tuple[float, float, float]:
    br, bz, bphi = _triple(rzphi)
    if plane == "xz":
        return (br, bphi, bz)
    return (br, bz, -bphi)


def _uniform_lab_to_rzphi(lab: Sequence[float], plane: str
                           ) -> tuple[float, float, float]:
    """Map the sampler's lab vector to physical cylindrical (r,z,phi)."""
    x, y, z = _triple(lab)
    if plane == "xz":
        return (x, z, y)
    return (x, y, -z)


def _parse_external_field(d: dict | None, geometry: str = "cartesian",
                          plane: str = "xy") -> Union[ExternalField, None]:
    if d is None:
        return None
    ev = _require(d, "evaluator", "external_field")
    if not isinstance(ev, dict):
        raise ValueError("external_field.evaluator must be a mapping")
    ev_type = str(_require(ev, "type", "external_field.evaluator"))
    _validate_evaluator_type(ev_type, "external_field.evaluator.type")

    def _reject_unknown(allowed: set[str]) -> None:
        unknown = sorted(
            (key for key in ev if key not in allowed), key=lambda key: str(key))
        if unknown:
            raise ValueError(
                "external_field.evaluator contains unsupported key(s) "
                f"{unknown} for {ev_type!r}")

    raw_conductors = ev.get("conductors", [])
    if not isinstance(raw_conductors, list):
        raise ValueError("external_field.evaluator.conductors must be a list")
    conductors = list(raw_conductors)

    if ev_type == "biot_savart":
        _reject_unknown({"type", "conductors"})
        return ExternalField(
            evaluator_type=ev_type, conductors=conductors)

    if ev_type == "uniform":
        _reject_unknown({
            "type", "B_T", "b_tesla", "B", "E_V_per_m",
            "e_v_per_m", "E", "B_rzphi"})
        # Uniform fields may use either terse or unit-explicit names.
        b = _unique_alias(
            ev, ("B_T", "b_tesla", "B"),
            "external_field.evaluator uniform B")
        e = _unique_alias(
            ev, ("E_V_per_m", "e_v_per_m", "E"),
            "external_field.evaluator uniform E")
        # Cylindrical decks may instead spell B by physical (r,z,phi) axes.
        b_rzphi = ev.get("B_rzphi")
        if b_rzphi is not None:
            if geometry != "cylindrical":
                raise ValueError(
                    "external_field.evaluator.B_rzphi (physical r,z,phi "
                    "axes) is only valid for geometry 'cylindrical'; use B_T "
                    "for cartesian decks")
            if b is not None:
                raise ValueError(
                    "external_field.evaluator: give either B_rzphi (physical "
                    "axes) or B_T (storage slots), not both")
            b = list(_rzphi_to_uniform_lab(
                b_rzphi, plane, "external_field.evaluator.B_rzphi"))
        return ExternalField(
            evaluator_type=ev_type, uniform_b=_vec3(b), uniform_e=_vec3(e))

    if ev_type == "dipole":
        _reject_unknown({
            "type", "moment_Am2", "moment_A_m2", "moment",
            "origin_xyz_m", "origin"})
        moment = _unique_alias(
            ev, ("moment_Am2", "moment_A_m2", "moment"),
            "external_field.evaluator dipole moment")
        origin = _unique_alias(
            ev, ("origin_xyz_m", "origin"),
            "external_field.evaluator origin", [0.0, 0.0, 0.0])
        return ExternalField(
            evaluator_type=ev_type,
            dipole_moment=None if moment is None else _vec3(moment),
            dipole_origin=_vec3(origin))

    if ev_type == "gradient":
        _reject_unknown({
            "type", "B0_T", "b0_tesla", "b0", "grad_T_per_m",
            "gradient_T_per_m", "gradient", "origin_xyz_m", "origin"})
        b0 = _unique_alias(
            ev, ("B0_T", "b0_tesla", "b0"),
            "external_field.evaluator gradient B0", [0.0, 0.0, 0.0])
        grad = _unique_alias(
            ev, ("grad_T_per_m", "gradient_T_per_m", "gradient"),
            "external_field.evaluator gradient matrix")
        origin = _unique_alias(
            ev, ("origin_xyz_m", "origin"),
            "external_field.evaluator origin", [0.0, 0.0, 0.0])
        return ExternalField(
            evaluator_type=ev_type, gradient_b0=_vec3(b0),
            gradient_matrix=_matrix3(grad), gradient_origin=_vec3(origin))

    if ev_type == "file_grid":
        _reject_unknown({"type", "path", "file"})
        file_path = _unique_alias(
            ev, ("path", "file"),
            "external_field.evaluator file-grid path")
        return ExternalField(
            evaluator_type=ev_type,
            file_path=None if file_path is None else str(file_path))

    # A registry plugin has no schema-specific aliases in the core package.
    # Its configure payload is therefore explicit and lossless.
    _reject_unknown({"type", "params", "conductors"})
    return ExternalField(
        evaluator_type=ev_type, conductors=conductors,
        params=_flat_evaluator_params(
            ev.get("params", {}), "external_field.evaluator.params"))


def _parse_species(items: list[dict] | None) -> list[Species]:
    if not items:
        return []
    out: list[Species] = []
    for raw in items:
        name = str(_require(raw, "name", "species"))
        init_raw = raw.get("initial", {})
        region_raw = init_raw.get("region", {}) or {}
        perturb_raw = init_raw.get("velocity_perturbation")
        perturbation = None
        if perturb_raw is not None:
            if not isinstance(perturb_raw, dict):
                raise ValueError(
                    f"species {name!r}: initial.velocity_perturbation must be "
                    "a mapping")
            unknown = set(perturb_raw) - {"amplitude_v", "mode", "phase_rad"}
            if unknown:
                raise ValueError(
                    f"species {name!r}: initial.velocity_perturbation contains "
                    f"unknown key(s): {sorted(unknown)}")
            mode_raw = perturb_raw.get("mode", (1, 0))
            if isinstance(mode_raw, (list, tuple)) and len(mode_raw) == 2:
                mode = (
                    _as_integer(
                        mode_raw[0],
                        f"species {name!r}: velocity_perturbation.mode[0]"),
                    _as_integer(
                        mode_raw[1],
                        f"species {name!r}: velocity_perturbation.mode[1]"))
            else:
                raise ValueError(
                    f"species {name!r}: initial.velocity_perturbation.mode must "
                    "be a two-element integer list")
            perturbation = VelocityPerturbation(
                amplitude_v=_vec3(_require(
                    perturb_raw, "amplitude_v",
                    f"species {name!r}: initial.velocity_perturbation")),
                mode=mode,
                phase_rad=float(perturb_raw.get("phase_rad", 0.0)))
        def _opt(key):
            v = region_raw.get(key)
            return None if v is None else float(v)
        initial = SpeciesInitial(
            distribution=str(init_raw.get("distribution", "maxwellian_uniform")),
            density_per_m3=float(init_raw.get("density_per_m3", 1.0e18)),
            temperature_eV=float(init_raw.get("temperature_eV", 1.0)),
            drift_v=_vec3(init_raw.get("drift_v")),
            velocity_perturbation=perturbation,
            region_x_min_m=_opt("x_min_m"),
            region_x_max_m=_opt("x_max_m"),
            region_y_min_m=_opt("y_min_m"),
            region_y_max_m=_opt("y_max_m"),
        )
        out.append(Species(
            name=name,
            charge_C=float(_require(raw, "charge_C", f"species {name!r}")),
            mass_kg=float(_require(raw, "mass_kg", f"species {name!r}")),
            n_particles=_as_integer(
                _require(raw, "n_particles", f"species {name!r}"),
                f"species {name!r}: n_particles"),
            initial=initial,
        ))
    return out


def _parse_time(d: dict | None) -> Time:
    if d is None:
        return Time()
    if not isinstance(d, dict):
        raise ValueError("time must be a mapping")
    unknown = set(d) - {"dt_s", "steps", "t_end_s"}
    if unknown:
        raise ValueError(f"time contains unknown key(s): {sorted(unknown)}")
    dt_raw = d.get("dt_s", "auto")
    dt_s: Union[float, str] = dt_raw if isinstance(dt_raw, str) else float(dt_raw)
    t_end_raw = d.get("t_end_s")
    return Time(dt_s=dt_s,
                steps=_as_integer(d.get("steps", 100), "time.steps"),
                t_end_s=(None if t_end_raw is None else float(t_end_raw)))


def _parse_diagnostics(d: dict | None) -> Diagnostics:
    if d is None:
        return Diagnostics()
    return Diagnostics(
        output_path=str(d.get("output_path", "out.npz")),
        cadence=_as_integer(d.get("cadence", 0), "diagnostics.cadence"),
        fields=list(d.get("fields", ["bz"])),
        per_species=_as_boolean(
            d.get("per_species", True), "diagnostics.per_species"),
    )


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
    if isinstance(mode_raw, int) and not isinstance(mode_raw, bool):
        mode = (_as_integer(mode_raw, "fields.initial.mode"), 0)
    else:
        if not isinstance(mode_raw, (list, tuple)) or len(mode_raw) not in (1, 2):
            raise ValueError("fields.initial.mode must be an integer or a 1/2-element list")
        mode = (
            _as_integer(mode_raw[0], "fields.initial.mode[0]"),
            _as_integer(mode_raw[1], "fields.initial.mode[1]")
            if len(mode_raw) == 2 else 0)
    raw_component = str(init.get(
        "component", "Ez" if geometry == "cylindrical" else "Ey"))
    return Fields(initial=FieldsInitial(
        # Resolve the physical/slot name to a storage slot here, once, so every
        # downstream consumer (cli._seed_fields) sees a plain slot name.
        type=str(_require(init, "type", "fields.initial")),
        component=_resolve_seed_component(raw_component, geometry),
        amplitude=float(init.get("amplitude", 1.0e-4)),
        mode=mode,
    ))


def parse(data: dict, *, base_dir: Path | str | None = None) -> PicDeck:
    """Parse a PIC deck, resolving confined file-grid data when requested.

    ``base_dir`` is mandatory only for a ``file_grid`` evaluator.  :func:`load`
    supplies the input deck's directory; callers parsing an in-memory mapping
    must state the intended filesystem boundary explicitly.
    """
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
        neutralizing_background=_as_boolean(
            data.get("neutralizing_background", False),
            "neutralizing_background"),
        raw=data,
    )
    deck.validate()
    if (deck.external_field is not None
            and deck.external_field.evaluator_type == "file_grid"):
        if base_dir is None:
            raise ValueError(
                "file_grid decks parsed from a mapping require parse(..., "
                "base_dir=...); use load(path) for YAML decks")
        deck.external_field.file_grid_params = load_file_grid_npz(
            base_dir, deck.external_field.file_path,
            label="external_field.evaluator.path")
    return deck


def load(path: Union[str, Path]) -> PicDeck:
    deck_path = Path(path).resolve()
    with open(deck_path) as fh:
        data = _load_yaml(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return parse(data, base_dir=deck_path.parent)
