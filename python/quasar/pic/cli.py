"""``quasar pic`` command-line entry point.

Subcommands:

* ``run <input.yaml>`` — build the solver, seed species, step, dump ``out.npz``.

Run with::

    python -m quasar.pic.cli run examples/square_toroid_pic/input.yaml
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import _core
from .._paths import confine_output_path, positive_int as _positive_int
from ..coil.io import build_conductor_system
from . import initial_conditions as ic
from . import io as pic_io
from ._units import QE as EV_TO_J  # elementary charge in C == eV->J factor
from ._units import Units
from .numerics import (besselj0, cfl_dt, cfl_limit, cyl_cfl_dt, cyl_cfl_limit,
                       j0_zero)


def _internal_spacing(domain, units: Units) -> tuple[float, float]:
    # The solver runs in internal units where c = 1; grid spacing enters the CFL
    # in those same internal lengths (the identity for a normalized deck).
    return (units.length(domain.lx_m) / domain.nx,
            units.length(domain.ly_m) / domain.ny)


def _cfl_dt_internal(domain, units: Units, fdtd_order: int = 2) -> float:
    dx, dy = _internal_spacing(domain, units)
    return cfl_dt(dx, dy, c=1.0, fdtd_order=fdtd_order)


def _cfl_limit_internal(domain, units: Units, fdtd_order: int = 2) -> float:
    dx, dy = _internal_spacing(domain, units)
    return cfl_limit(dx, dy, c=1.0, fdtd_order=fdtd_order)


def _cyl_cfl_dt_internal(domain, units: Units, fdtd_order: int = 2) -> float:
    dr, dz = _internal_spacing(domain, units)
    return cyl_cfl_dt(dr, dz, c=1.0, fdtd_order=fdtd_order)


def _cyl_cfl_limit_internal(domain, units: Units, fdtd_order: int = 2) -> float:
    dr, dz = _internal_spacing(domain, units)
    return cyl_cfl_limit(dr, dz, c=1.0, fdtd_order=fdtd_order)


def _required_solver_nghost(deck: pic_io.PicDeck) -> int:
    """Minimum halo shared by solver construction and test-double fallbacks."""
    return max(_core.pic.required_nghost(deck.numerics.fdtd_order),
               2 if deck.numerics.shape == "tsc" else 1)


def _solver_nghost(solver, deck: pic_io.PicDeck) -> int:
    """Return the constructed solver's authoritative padded-grid halo."""
    accessor = getattr(solver, "nghost", None)
    if accessor is None:
        # Lightweight unit-test doubles predate the bound grid accessor. Keep
        # their fallback on the same construction helper rather than duplicating
        # the FDTD/shape rule at each call site.
        return _required_solver_nghost(deck)
    value = accessor() if callable(accessor) else accessor
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("solver nghost must be an integer")
    value = int(value)
    if value < _required_solver_nghost(deck):
        raise ValueError("solver halo is too small for the selected numerics")
    return value


def _macro_weight(configured_density: float, internal_density: float,
                  particle_volume: float, species_name: str) -> float:
    """Form density*volume without false intermediate under/overflow."""
    if configured_density == 0.0:
        return 0.0
    if not (math.isfinite(internal_density) and internal_density > 0.0):
        raise OverflowError(
            f"species {species_name!r}: positive density is not representable "
            "in internal units")
    if not (math.isfinite(particle_volume) and particle_volume > 0.0):
        raise OverflowError(
            f"species {species_name!r}: represented particle volume is not "
            "finite and positive")
    density_mantissa, density_exponent = math.frexp(internal_density)
    volume_mantissa, volume_exponent = math.frexp(particle_volume)
    mantissa, adjustment = math.frexp(density_mantissa * volume_mantissa)
    try:
        result = math.ldexp(
            mantissa, density_exponent + volume_exponent + adjustment)
    except OverflowError as exc:
        raise OverflowError(
            f"species {species_name!r}: macro-particle weight is not "
            "representable") from exc
    if not (math.isfinite(result) and result > 0.0):
        raise OverflowError(
            f"species {species_name!r}: positive density produces an "
            "unrepresentable macro-particle weight")
    return result


def _make_solver(deck: pic_io.PicDeck, units: Units):
    pic = _core.pic
    # Grid coordinates enter the solver in internal length units. The ghost halo
    # must be wide enough for the FDTD order (order 4 reads two cells past a
    # wall) and for TSC's first true ghost beyond a physical high Yee face.
    nghost = _required_solver_nghost(deck)
    grid = pic.Grid2D(nx=deck.domain.nx, ny=deck.domain.ny,
                      lx=units.length(deck.domain.lx_m),
                      ly=units.length(deck.domain.ly_m),
                      origin_x=units.length(deck.domain.origin_x_m),
                      origin_y=units.length(deck.domain.origin_y_m),
                      nghost=nghost)
    cfg = pic.EmPicConfig()
    cfg.grid = grid
    cfg.fdtd_order = deck.numerics.fdtd_order
    cfg.shape = deck.numerics.shape
    cfg.plane = deck.plane
    cfg.geometry = deck.geometry
    cfg.neutralizing_background = deck.neutralizing_background
    if units.normalization is not None:
        cfg.normalization = units.normalization
    # Boundaries are selected by registry name; the deck strings (already
    # validated in io.py) pass straight through to the C++ registry.
    for side, name in enumerate(deck.boundary.particle):
        cfg.boundary.set_particle_side(side, name)
    for side, name in enumerate(deck.boundary.field):
        cfg.boundary.set_field_side(side, name)
    # Current-smoothing pipeline: each entry is {type: <name>, n_passes|passes}.
    cfg.filters = [
        pic.FilterSpec(name=str(spec["type"]),
                       passes=int(spec.get("n_passes", spec.get("passes", 1))))
        for spec in deck.numerics.current_filter
    ]
    return pic.EmPic2D3V(cfg)


def _seed_species(solver, deck: pic_io.PicDeck, units: Units,
                  rng: np.random.Generator) -> list[int]:
    pic = _core.pic
    indices: list[int] = []
    for sp in deck.species:
        # In normalized decks temperature is a dimensionless kinetic energy in
        # the same mass/velocity system as the solver, so v_th=sqrt(T/m). Applying
        # the SI eV->J formula to identity-unit values suppresses v_th by ~1e-10.
        if units.identity:
            v_thermal = float(np.sqrt(sp.initial.temperature_eV / sp.mass_kg))
        else:
            v_thermal_si = float(
                np.sqrt(sp.initial.temperature_eV * EV_TO_J / sp.mass_kg))
            v_thermal = units.velocity(v_thermal_si)
        drift = tuple(units.velocity(c) for c in sp.initial.drift_v)
        domain_lx = units.length(deck.domain.lx_m)
        domain_ly = units.length(deck.domain.ly_m)
        domain_ox = units.length(deck.domain.origin_x_m)
        domain_oy = units.length(deck.domain.origin_y_m)
        if sp.initial.distribution == "maxwellian_block":
            x_min = units.length(sp.initial.region_x_min_m)
            x_max = units.length(sp.initial.region_x_max_m)
            y_min = units.length(sp.initial.region_y_min_m)
            y_max = units.length(sp.initial.region_y_max_m)
            if deck.geometry == "cylindrical":
                positions = ic.quiet_positions_rz_block(
                    sp.n_particles, x_min, x_max, y_min, y_max)
                particle_volume = ic.quiet_block_ring_volume(
                    sp.n_particles, x_min, x_max, y_min, y_max)
            else:
                positions = ic.quiet_positions_2d_block(
                    sp.n_particles, x_min, x_max, y_min, y_max)
                particle_volume = ic.quiet_block_cell_area(
                    sp.n_particles, x_min, x_max, y_min, y_max)
        else:
            if deck.geometry == "cylindrical":
                positions = ic.quiet_positions_rz_block(
                    sp.n_particles, domain_ox, domain_ox + domain_lx,
                    domain_oy, domain_oy + domain_ly)
                particle_volume = ic.quiet_block_ring_volume(
                    sp.n_particles, domain_ox, domain_ox + domain_lx,
                    domain_oy, domain_oy + domain_ly)
            else:
                positions = ic.quiet_positions_2d_block(
                    sp.n_particles, domain_ox, domain_ox + domain_lx,
                    domain_oy, domain_oy + domain_ly)
                particle_volume = ic.quiet_block_cell_area(
                    sp.n_particles, domain_ox, domain_ox + domain_lx,
                    domain_oy, domain_oy + domain_ly)
        velocities = ic.maxwellian(sp.n_particles, v_thermal,
                                   drift=drift,
                                   seed=int(rng.integers(0, 2**31 - 1)))
        perturbation = sp.initial.velocity_perturbation
        if perturbation is not None:
            mx, my = perturbation.mode
            reduced_phase = math.remainder(
                perturbation.phase_rad, 2.0 * math.pi)
            phase = (2.0 * np.pi
                     * (mx * (positions[:, 0] - domain_ox) / domain_lx
                        + my * (positions[:, 1] - domain_oy) / domain_ly)
                     + reduced_phase)
            amplitude = np.asarray(
                [units.velocity(value) for value in perturbation.amplitude_v],
                dtype=float)
            velocities += np.sin(phase)[:, np.newaxis] * amplitude
        speeds = np.linalg.norm(velocities, axis=1)
        if speeds.size and float(np.max(speeds)) >= 1.0:
            raise ValueError(
                f"species {sp.name!r}: sampled |v|/c >= 1, outside the "
                "nonrelativistic Boris model; lower temperature/drift or use a "
                "relativistic pusher")

        # The rank-1 quiet start partitions the full Cartesian area (or
        # cylindrical volume) equally among all particles, so each macro-weight
        # is density times that per-particle area/volume.  Block initializers use
        # the corresponding selected-block area/volume divided by their count.
        macro_weight = _macro_weight(
            sp.initial.density_per_m3,
            units.density(sp.initial.density_per_m3), particle_volume, sp.name)
        weights = np.full(sp.n_particles, macro_weight)

        cfg = pic.SpeciesConfig(name=sp.name,
                                charge=units.charge(sp.charge_C),
                                mass=units.mass(sp.mass_kg),
                                capacity=sp.n_particles)
        idx = solver.add_species(cfg)
        # These samples are the physical velocity distribution at t=0. The C++
        # solver owns the initial dt/2 Boris kick; the deck path must not
        # pre-stagger them. The pybind binding is declared with
        # py::array::forcecast, so it copies to contiguous float64 itself; no
        # host-side astype needed here.
        solver.set_species_particles(
            idx,
            x=positions[:, 0],
            y=positions[:, 1],
            vx=velocities[:, 0],
            vy=velocities[:, 1],
            vz=velocities[:, 2],
            weight=weights,
        )
        indices.append(idx)
    return indices


def _apply_external_field(solver, deck: pic_io.PicDeck, units: Units) -> None:
    if deck.external_field is None:
        return
    ms = _core.magnetostatics
    ef = deck.external_field
    # Build the evaluator purely by registry name, then push the deck parameters
    # through the uniform configure() seam — no per-type branch. Each evaluator
    # validates the keys it knows (e.g. uniform accepts b0/e0), so misspelled or
    # unsupported parameters fail rather than being silently ignored.
    evaluator = ms.create_field_evaluator(ef.evaluator_type)
    evaluator.configure(ef.evaluator_params())
    cs = build_conductor_system(ef.conductors)
    length_scale, e_field_scale, b_field_scale = units.external_scales()
    solver.sample_external_field(
        evaluator, cs, length_scale, e_field_scale, b_field_scale,
        plane=deck.plane)


def _seed_fields(solver, deck: pic_io.PicDeck, dt: float = 0.0,
                 units: Units | None = None) -> None:
    """Apply an optional initial field seed.

    The interior is written on a ghost-padded buffer matching the solver storage
    (pitch = nx + 2*nghost). SI amplitudes are interpreted as V/m for an E
    component and tesla for a B component, then converted to the solver's natural
    units. Normalized-deck amplitudes pass through unchanged."""
    init = deck.fields.initial
    if init is None:
        return
    if units is None:
        units = Units(deck)
    nx, ny = deck.domain.nx, deck.domain.ny
    # The constructed grid is authoritative: shape support can require a wider
    # halo than the curl alone (notably order-two TSC).
    g = _solver_nghost(solver, deck)
    pitch = nx + 2 * g
    height = ny + 2 * g

    if deck.geometry == "cylindrical":
        offsets = {
            "ex": (0.0, 0.5), "ey": (0.5, 0.0), "ez": (0.0, 0.5),
            "bx": (0.0, 0.0), "by": (0.5, 0.5), "bz": (0.0, 0.0),
        }
    else:
        offsets = {
            "ex": (0.0, 0.5), "ey": (0.5, 0.0), "ez": (0.5, 0.5),
            "bx": (0.5, 0.0), "by": (0.0, 0.5), "bz": (0.0, 0.0),
        }

    def _lattice_shape(component: str) -> tuple[int, int, float, float]:
        ox, oy = offsets[component]
        return nx + (ox == 0.0), ny + (oy == 0.0), ox, oy

    def _seed(component: str, interior: np.ndarray) -> None:
        # Write every physical degree of freedom on the component's own Yee
        # lattice. Face-located nonperiodic components include their independent
        # high face at index nx/ny; periodic fills later make that face duplicate.
        x_count, y_count, _, _ = _lattice_shape(component)
        interior = np.asarray(interior, dtype=np.float64)
        if interior.shape != (y_count, x_count):
            raise ValueError(
                f"seed profile for {component} has shape {interior.shape}; "
                f"expected {(y_count, x_count)}")
        buf = np.zeros((height, pitch), dtype=np.float64)
        buf[g:g + y_count, g:g + x_count] = interior
        solver.seed_field(component, buf.reshape(-1))

    comp = init.component.lower()
    seed_amplitude = (units.e_field(init.amplitude)
                      if comp.startswith("e")
                      else units.b_field(init.amplitude))
    if (not math.isfinite(seed_amplitude)
            or (init.amplitude != 0.0 and seed_amplitude == 0.0)):
        unit_name = "V/m" if comp.startswith("e") else "tesla"
        raise OverflowError(
            f"fields.initial.amplitude in {unit_name} is not representable in "
            "the solver normalization")

    def _sinusoid(component: str, phase_shift: float = 0.0) -> np.ndarray:
        x_count, y_count, ox, _ = _lattice_shape(component)
        x_index = np.arange(x_count, dtype=float) + ox
        row = seed_amplitude * np.sin(
            2.0 * np.pi * init.mode[0] * x_index / nx + phase_shift)
        return np.broadcast_to(row, (y_count, x_count)).copy()

    def _staggered_numerator(half_cell_phase: float) -> float:
        """Dimensionless numerator of the centre/face Yee derivative symbol."""
        if deck.numerics.fdtd_order == 4:
            return ((9.0 / 4.0) * math.sin(half_cell_phase)
                    - (1.0 / 12.0) * math.sin(3.0 * half_cell_phase))
        return 2.0 * math.sin(half_cell_phase)

    if init.type == "seed_perturbation":
        mx = max(1, init.mode[0])
        if deck.geometry == "cylindrical":
            if comp != "ey" or deck.domain.origin_x_m != 0.0:
                raise ValueError(
                    "cylindrical seed_perturbation supports only on-axis "
                    "physical Ez (storage component Ey)")
            if deck.boundary.field[1] != "pec":
                raise ValueError(
                    "cylindrical seed_perturbation requires a PEC "
                    "outer-radius field boundary")
            if mx > nx:
                raise ValueError(
                    "cylindrical seed mode exceeds the resolved radial "
                    "Nyquist spectrum")
            # On a cylindrical (r,z) grid the radial axis is not a plain Cartesian
            # interval: a sin(2 pi r/R) perturbation is dominated by higher radial
            # Bessel modes (it has ~90% overlap with TM020, only ~3% with TM010),
            # so the cavity would ring at the wrong line. Seed the physically
            # appropriate axisymmetric radial eigenmode J0(j_{0,mx} r/R) instead,
            # which excites the TM0,mx,0 mode cleanly (mx=1 -> TM010). Uniform in z.
            # j0_zero / besselj0 are dependency-free (numpy only) so a cylindrical
            # deck does not pull in scipy.
            j0n = j0_zero(mx)
            x_count, y_count, ox, _ = _lattice_shape(comp)
            r_over_R = (np.arange(x_count, dtype=float) + ox) / nx
            row = seed_amplitude * besselj0(j0n * r_over_R)

            # The solver stores B at t=-dt/2 while E is at t=0.  A standing
            # cavity mode is time-symmetric about t=0, so Faraday requires
            # Bphi^- = -(dt/2) D_r^- Ez.  Leaving Bphi at zero instead gives a
            # half-step phase lead and an O((omega*dt)^2) amplitude error.
            if not (math.isfinite(dt) and dt > 0.0):
                raise ValueError(
                    "cylindrical seed_perturbation requires the positive "
                    "solver timestep so Bphi can be initialized at t=-dt/2")
            dr, _ = _internal_spacing(deck.domain, units)

            def radial_ez(cell: int) -> float:
                # Axis regularity is even for Ez.  The J0 root lies at the
                # outer PEC wall, where tangential cell-centred Ez has the odd
                # continuation used by the live boundary kernel.
                sign = 1.0
                while cell < 0 or cell >= nx:
                    if cell < 0:
                        cell = -cell - 1
                    else:
                        cell = 2 * nx - cell - 1
                        sign = -sign
                return sign * float(row[cell])

            derivative = np.empty(nx + 1, dtype=np.float64)
            for face in range(nx + 1):
                if deck.numerics.fdtd_order == 4:
                    derivative[face] = (
                        (9.0 / 8.0)
                        * (radial_ez(face) - radial_ez(face - 1))
                        - (1.0 / 24.0)
                        * (radial_ez(face + 1) - radial_ez(face - 2))) / dr
                else:
                    derivative[face] = (
                        radial_ez(face) - radial_ez(face - 1)) / dr
            bphi_row = -0.5 * dt * derivative
            bx_count, by_count, _, _ = _lattice_shape("bz")
            _seed(comp, np.broadcast_to(row, (y_count, x_count)).copy())
            _seed("bz", np.broadcast_to(
                bphi_row, (by_count, bx_count)).copy())
        else:
            if comp in ("ex", "bx"):
                raise ValueError(
                    "cartesian seed_perturbation cannot seed longitudinal "
                    "Ex or Bx because that would violate Gauss's law or div(B)=0")
            if 2 * mx > nx:
                raise ValueError("seed mode exceeds the resolved x Nyquist mode")
            _seed(comp, _sinusoid(comp))
    elif init.type == "seed_tm_cavity":
        if deck.geometry != "cartesian":
            raise ValueError(
                "seed_tm_cavity is defined only for Cartesian rectangular "
                "cavities")
        if comp != "ez":
            raise ValueError(
                "seed_tm_cavity component must be Ez (the out-of-plane "
                "electric field)")
        if any(name != "pec" for name in deck.boundary.field):
            raise ValueError(
                "seed_tm_cavity requires PEC field boundaries on all four sides")
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(
                "seed_tm_cavity requires the positive solver timestep so B can "
                "be initialized at t=-dt/2")

        mx, my = init.mode
        if mx < 1 or my < 1 or mx > nx or my > ny:
            raise ValueError(
                "seed_tm_cavity mode must lie in [1,nx] x [1,ny]")
        dx, dy = _internal_spacing(deck.domain, units)
        numerator_x = _staggered_numerator(math.pi * mx / (2.0 * nx))
        numerator_y = _staggered_numerator(math.pi * my / (2.0 * ny))
        # Form dt*kappa directly, never kappa~1/h.  A representable CFL-safe
        # step can multiply an individually unrepresentable modified wavenumber
        # on an extremely fine grid.  The direction ratios use h_min/h so their
        # hypot arguments remain bounded even for extreme aspect ratios.
        phase_x = numerator_x * (dt / dx)
        phase_y = numerator_y * (dt / dy)
        phase_norm = math.hypot(phase_x, phase_y)
        dispersion_arg = 0.5 * phase_norm
        if dispersion_arg > 1.0 + 64.0 * np.finfo(float).eps:
            raise ValueError(
                "seed_tm_cavity dt is outside the discrete Maxwell stability "
                "branch")
        omega_dt = 2.0 * math.asin(min(1.0, dispersion_arg))
        h_min = min(dx, dy)
        direction_x = numerator_x * (h_min / dx)
        direction_y = numerator_y * (h_min / dy)
        direction_norm = math.hypot(direction_x, direction_y)

        # TM_mn on the exact Cartesian Yee lattices.  Odd continuation of the
        # cell-centred Ez sine at all four walls makes its wall interpolation
        # exactly zero.  Bx (x-centred/y-face) and By (x-face/y-centred) have the
        # matching sine/cosine parity, hence discrete div(B)=0 at every Yee node.
        sx = np.sin(math.pi * mx * (np.arange(nx, dtype=float) + 0.5) / nx)
        sy = np.sin(math.pi * my * (np.arange(ny, dtype=float) + 0.5) / ny)
        cx = np.cos(math.pi * mx * np.arange(nx + 1, dtype=float) / nx)
        cy = np.cos(math.pi * my * np.arange(ny + 1, dtype=float) / ny)
        half_time = math.sin(0.5 * omega_dt)

        _seed("ez", seed_amplitude * np.multiply.outer(sy, sx))
        _seed("bx", (seed_amplitude * (direction_y / direction_norm) * half_time
                     * np.multiply.outer(cy, sx)))
        _seed("by", (-seed_amplitude * (direction_x / direction_norm) * half_time
                     * np.multiply.outer(sy, cx)))
    elif init.type == "seed_em_wave":
        mx, my = init.mode
        # Only +x propagation is implemented; a non-zero my would request a +y /
        # oblique wave that this seeding does not build, so reject it loudly rather
        # than silently seeding a +x wave instead.
        if my != 0:
            raise ValueError(
                f"seed_em_wave: only +x propagation (mode my=0) is supported, "
                f"got mode={init.mode!r}")
        if deck.geometry == "cylindrical":
            raise ValueError(
                "seed_em_wave is invalid for cylindrical geometry: a radially "
                "translating Cartesian plane wave is not a regular axisymmetric "
                "mode; use an axial seed_perturbation Bessel cavity mode")
        if 2 * mx >= nx:
            raise ValueError(
                "seed_em_wave mode must lie strictly below the resolved x "
                "Nyquist mode")
        # A +x-propagating leapfrog wave. E is stored at t=0 on x-centres while
        # B is stored at t=-dt/2 on x-faces; seeding both at the same phase/time
        # injects a first-order temporal error before the first update.
        theta = np.pi * mx / nx  # k*dx/2
        # ``dt`` is already in the solver's normalized time units.  Use the
        # matching normalized cell spacing in the discrete dispersion relation;
        # mixing it with the SI ``lx_m`` from an SI deck gives both a wrong
        # magnetic half-step phase and a dimensionally meaningless CFL check.
        dx = units.length(deck.domain.lx_m) / nx
        dispersion_arg = 0.5 * _staggered_numerator(theta) * (dt / dx)
        if abs(dispersion_arg) > 1.0 + 64.0 * np.finfo(float).eps:
            raise ValueError("seed_em_wave dt is outside the discrete CFL branch")
        omega_dt = 2.0 * np.arcsin(np.clip(dispersion_arg, -1.0, 1.0))
        if comp == "ez":
            _seed("ez", _sinusoid("ez"))
            _seed("by", -_sinusoid("by", 0.5 * omega_dt))
        elif comp == "ey":
            _seed("ey", _sinusoid("ey"))
            _seed("bz", _sinusoid("bz", 0.5 * omega_dt))
        else:
            raise ValueError(f"seed_em_wave: unsupported component {init.component!r}")
    else:
        raise ValueError(f"fields.initial.type {init.type!r} is not supported")


def _species_to_si(host: dict, units: Units) -> dict:
    # Positions are lengths; vx/vy/vz are velocities. weight/alive are unitless.
    out = dict(host)
    out["x"] = units.length_to_si(host["x"])
    out["y"] = units.length_to_si(host["y"])
    out["vx"] = units.velocity_to_si(host["vx"])
    out["vy"] = units.velocity_to_si(host["vy"])
    out["vz"] = units.velocity_to_si(host["vz"])
    return out


def _snapshot(solver, deck: pic_io.PicDeck, species_indices: list[int],
              step: int, sim_time: float, units: Units) -> dict:
    # The solver copies only the requested components host-side, one device->host
    # transfer per component rather than the whole six-field dict.
    snap = {
        "step": step,
        "time_s": sim_time,
        "fields": {k: units.field_component_to_si(
                       k, solver.field_component_to_host(k))
                   for k in deck.diagnostics.fields},
        "external_bx": units.field_component_to_si(
            "bx", solver.external_field_component_to_host("bx")),
        "external_by": units.field_component_to_si(
            "by", solver.external_field_component_to_host("by")),
        "external_bz": units.field_component_to_si(
            "bz", solver.external_field_component_to_host("bz")),
        "nx": deck.domain.nx,
        "ny": deck.domain.ny,
        # The Yee buffers are ghost-padded; persist the halo width so the offline
        # reader strips the right number of cells instead of re-deriving it from
        # the flat size.
        "nghost": _solver_nghost(solver, deck),
        # Which lab plane the grid represents, so offline readers can label the
        # in-plane axes and know that external_b{x,y,z} are PIC-frame components
        # (in "xz" mode the out-of-plane B_phi lands in external_bz).
        "plane": deck.plane,
        # Coordinate system of the grid so offline readers know whether to treat
        # the in-plane axes as (x,y) or axisymmetric r-z.
        "geometry": deck.geometry,
        # Persist the four field-side boundary kinds.  Yee components with a
        # zero offset own an independent high face on a wall axis, whereas the
        # same slot is a duplicate of the low face on a periodic axis.  An
        # offline reader cannot distinguish those cases from values alone.
        "boundary_field": tuple(deck.boundary.field),
        # Spatial metadata uses the same output coordinate units as particle
        # snapshots: metres for SI decks, identity/internal lengths otherwise.
        "origin_x": deck.domain.origin_x_m,
        "origin_y": deck.domain.origin_y_m,
        "lx": deck.domain.lx_m,
        "ly": deck.domain.ly_m,
        "unit_system": deck.units,
    }
    if deck.diagnostics.per_species:
        per_sp = {}
        for idx, sp in zip(species_indices, deck.species):
            per_sp[sp.name] = _species_to_si(solver.species_at(idx).to_host(), units)
        snap["species"] = per_sp
    return snap


def _flatten_for_npz(snapshots: list[dict], final: dict,
                     scalar_series: dict[str, list] | None = None,
                     ) -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {
        "final_step": np.array([final["step"]]),
        "final_time_s": np.array([final["time_s"]]),
        "nx": np.array([final["nx"]]),
        "ny": np.array([final["ny"]]),
        "nghost": np.array([final["nghost"]]),
        "plane": np.array([final.get("plane", "xy")]),
        "geometry": np.array([final.get("geometry", "cartesian")]),
        "boundary_field": np.asarray(final["boundary_field"], dtype=np.str_),
        "origin_x": np.array([final.get("origin_x", 0.0)]),
        "origin_y": np.array([final.get("origin_y", 0.0)]),
        "lx": np.array([final.get("lx", final["nx"])]),
        "ly": np.array([final.get("ly", final["ny"])]),
        "unit_system": np.array([final.get("unit_system", "SI")]),
        "external_bx": final["external_bx"],
        "external_by": final["external_by"],
        "external_bz": final["external_bz"],
    }
    for name, arr in final["fields"].items():
        flat[f"field_{name}"] = arr
    if "species" in final:
        flat["species_names"] = np.asarray(list(final["species"]), dtype=np.str_)
        for sp_name, host in final["species"].items():
            for k, v in host.items():
                flat[f"species_{sp_name}_{k}"] = v
    if snapshots:
        flat["snapshot_steps"] = np.array([s["step"] for s in snapshots])
        flat["snapshot_times_s"] = np.array([s["time_s"] for s in snapshots])
        for fname in final["fields"].keys():
            flat[f"snapshot_field_{fname}"] = np.stack(
                [s["fields"][fname] for s in snapshots])
    if scalar_series:
        for k, v in scalar_series.items():
            flat[f"series_{k}"] = np.asarray(v)
    return flat


def _indexed_output_path(base: Path, step: int, *, width: int = 10) -> Path:
    """Per-step output path: insert a zero-padded step index before the suffix.

    ``out.npz`` at step 10 -> ``out_0000000010.npz``. Only the stem is
    rewritten, so the parent directory and suffix (and thus the deck-directory
    confinement of ``base``) are preserved.
    """
    return base.with_name(f"{base.stem}_{step:0{width}d}{base.suffix}")


def _internal_end_time(deck: pic_io.PicDeck, units: Units) -> float | None:
    """Return the requested endpoint in solver units, if representable."""
    if deck.time.t_end_s is None:
        return None
    value = float(units.time(float(deck.time.t_end_s)))
    if not np.isfinite(value) or not value > 0.0:
        raise RuntimeError(
            "PIC run loop endpoint is not representable in solver units")
    return value


def prepare_run(deck: pic_io.PicDeck, units: Units, *, seed: int = 0):
    """Build a fully seeded solver from a parsed deck, ready to ``step(dt)``.

    Constructs the solver, applies the external field and any field seed, seeds
    all species, and resolves the internal/SI timestep. Returns
    ``(solver, species_indices, dt, dt_si)``. This is the shared build path for
    the ``run`` subcommand and for offline drivers (e.g. example plot scripts)
    so the solver-construction + seeding logic lives in one place."""
    solver = _make_solver(deck, units)
    _apply_external_field(solver, deck, units)

    # The solver steps in internal time units; an explicit deck dt_s is SI and is
    # converted, while "auto" is derived directly on the internal grid (c = 1).
    cylindrical = deck.geometry == "cylindrical"
    if deck.time.dt_s == "auto":
        if cylindrical:
            dt = _cyl_cfl_dt_internal(
                deck.domain, units, deck.numerics.fdtd_order)
        else:
            dt = _cfl_dt_internal(deck.domain, units, deck.numerics.fdtd_order)
    else:
        dt = units.time(float(deck.time.dt_s))
        # An explicit dt above the Yee CFL limit makes the FDTD update diverge
        # exponentially; the 'auto' path is CFL-safe by construction but a
        # user-supplied dt must be checked against the same limit.
        if cylindrical:
            cfl = _cyl_cfl_limit_internal(
                deck.domain, units, deck.numerics.fdtd_order)
            if dt > cfl:
                raise ValueError(
                    f"time.dt_s ({units.time_to_si(dt):.6e} s) exceeds the "
                    f"cylindrical (r-z) CFL stability limit "
                    f"({units.time_to_si(cfl):.6e} s) for this grid; reduce "
                    f"dt_s or use 'auto'."
                )
        else:
            cfl = _cfl_limit_internal(deck.domain, units, deck.numerics.fdtd_order)
            if dt > cfl:
                raise ValueError(
                    f"time.dt_s ({units.time_to_si(dt):.6e} s) exceeds the CFL "
                    f"stability limit ({units.time_to_si(cfl):.6e} s) for this grid "
                    f"and fdtd_order={deck.numerics.fdtd_order}; reduce dt_s or use "
                    f"'auto'."
                )
    if not np.isfinite(dt) or not dt > 0.0:
        raise ValueError(
            "time.dt_s is not representable as a positive solver timestep")
    dt_si = units.time_to_si(dt)

    # Leapfrog field seeds place B at the half time immediately preceding the
    # first E update.  If t_end clips that very first update, seed against the
    # clipped width rather than the nominal cadence; subsequent variable-step
    # centering is handled by EmPic2D3V::step.
    t_end_internal = _internal_end_time(deck, units)
    first_dt = dt if t_end_internal is None else min(dt, t_end_internal)
    _seed_fields(solver, deck, first_dt, units)
    rng = np.random.default_rng(seed)
    species_indices = _seed_species(solver, deck, units, rng)
    return solver, species_indices, dt, dt_si


def _do_run(args: argparse.Namespace) -> int:
    deck_path = Path(args.input).resolve()
    deck = pic_io.load(deck_path)
    if args.steps_override is not None:
        deck.time = pic_io.Time(dt_s=deck.time.dt_s,
                                steps=args.steps_override,
                                t_end_s=deck.time.t_end_s)
        deck.validate()

    units = Units(deck)
    solver, species_indices, dt, dt_si = prepare_run(deck, units, seed=args.seed)
    if args.print_config:
        print(f"deck   : {deck_path}")
        print(f"grid   : {deck.domain.nx}x{deck.domain.ny}  "
              f"({deck.domain.lx_m}x{deck.domain.ly_m}) m  "
              f"origin=({deck.domain.origin_x_m}, {deck.domain.origin_y_m})")
        print(f"units  : {deck.units}")
        print(f"plane  : {deck.plane}")
        print(f"geometry: {deck.geometry}")
        print(f"species: {[sp.name for sp in deck.species]}")
        end = ("" if deck.time.t_end_s is None
               else f"    t_end: {deck.time.t_end_s:.6e} s")
        print(f"dt     : {dt_si:.6e} s    steps: {deck.time.steps}{end}")

    # Confine the deck-supplied output path to the deck's own directory so a
    # stray absolute path or "../" cannot write outside it.
    out_path = confine_output_path(deck_path.parent, deck.diagnostics.output_path,
                                   label="diagnostics.output_path")
    _run_loop(solver, deck, species_indices, units, dt, dt_si, out_path, args)
    if args.print_config or args.verbose:
        print(f"wrote  : {out_path}")
    return 0


def _run_loop(solver, deck: pic_io.PicDeck, species_indices: list[int],
              units: Units, dt: float, dt_si: float, out_path,
              args: argparse.Namespace) -> None:
    snapshots: list[dict] = []
    sim_time = 0.0
    series: dict[str, list] = {"step": [], "time_s": []}
    for sp in deck.species:
        series[f"alive_{sp.name}"] = []
    t0 = time.time()

    def _record_scalars(step_done: int, t_now: float) -> None:
        series["step"].append(step_done)
        series["time_s"].append(t_now)
        for idx, sp in zip(species_indices, deck.species):
            # Device-side reduction; avoids a full 7-array host copy per logged step.
            series[f"alive_{sp.name}"].append(int(solver.species_alive_count(idx)))

    def _flush(step_done: int, t_now: float) -> None:
        final = _snapshot(solver, deck, species_indices, step_done, t_now, units)
        with out_path.open("wb") as stream:
            np.savez(stream, **_flatten_for_npz(snapshots, final, series))

    log_every = max(0, int(args.log_every))
    write_every = max(0, int(args.write_every))

    # Track the clock in the same internal units as the solver.  Clipping an SI
    # remainder and converting it back can differ by an ulp from clipping against
    # units.time(t_end_s), leaving the field/particle state at a time that does not
    # match its exact output label.  SI time is derived from this solver clock for
    # ordinary steps; the requested endpoint label is assigned exactly after the
    # final internal-domain clip.
    step_done = 0
    t_end = (None if deck.time.t_end_s is None
             else float(deck.time.t_end_s))
    solver_time = 0.0
    t_end_internal = _internal_end_time(deck, units)
    while step_done < deck.time.steps and (
            t_end_internal is None or solver_time < t_end_internal):
        dt_step = dt
        final_clipped_step = False
        if t_end_internal is not None:
            remaining_internal = t_end_internal - solver_time
            dt_step = min(dt_step, remaining_internal)
            final_clipped_step = dt_step == remaining_internal
        if not np.isfinite(dt_step) or not dt_step > 0.0:
            raise RuntimeError("PIC run loop timestep made no forward progress")
        next_solver_time = solver_time + dt_step
        if not next_solver_time > solver_time:
            raise RuntimeError("PIC run loop time made no floating-point progress")
        solver.step(dt_step)
        solver_time = (t_end_internal if final_clipped_step
                       else next_solver_time)
        sim_time = (t_end if final_clipped_step
                    else float(units.time_to_si(solver_time)))
        if not np.isfinite(sim_time):
            raise RuntimeError("PIC run loop SI time is not finite")
        step_done += 1
        if deck.diagnostics.cadence > 0 and step_done % deck.diagnostics.cadence == 0:
            snapshots.append(_snapshot(solver, deck, species_indices,
                                       step_done, sim_time, units))
        if log_every > 0 and step_done % log_every == 0:
            _record_scalars(step_done, sim_time)
            elapsed = time.time() - t0
            rate = step_done / elapsed if elapsed > 0 else 0.0
            remaining_steps = deck.time.steps - step_done
            if t_end is not None and dt_si > 0.0:
                remaining_steps = min(
                    remaining_steps,
                    max(0, int(np.ceil((t_end - sim_time) / dt_si))))
            remaining = remaining_steps / rate if rate > 0 else float("nan")
            alive_str = " ".join(f"{sp.name}={series[f'alive_{sp.name}'][-1]}"
                                  for sp in deck.species)
            print(f"step {step_done}/{deck.time.steps}  "
                  f"t={sim_time:.6e}s  "
                  f"rate={rate:.0f} step/s  "
                  f"eta={remaining:.0f}s  alive: {alive_str}",
                  flush=True)
        if write_every > 0 and step_done % write_every == 0:
            final = _snapshot(solver, deck, species_indices, step_done, sim_time, units)
            per_path = _indexed_output_path(out_path, step_done)
            with per_path.open("wb") as stream:
                np.savez(stream, **_flatten_for_npz([], final, None))

    # step() drains deposit failures before Ampere. Keep the public final drain as
    # a defensive check for alternate/low-level integrations.
    solver.finalize()

    if log_every == 0 or step_done % log_every != 0:
        _record_scalars(step_done, sim_time)
    _flush(step_done, sim_time)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quasar-pic",
                                description="Quasar PIC driver.")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a PIC simulation from a YAML deck.")
    run.add_argument("input", help="Path to the YAML deck.")
    run.add_argument("--seed", type=int, default=0,
                     help="RNG seed for initial-condition sampling.")
    run.add_argument("--verbose", action="store_true",
                     help="print informational output (default: quiet)")
    run.add_argument("--print-config", action="store_true",
                     help="Print resolved deck + dt before running.")
    run.add_argument("--steps-override", type=_positive_int, default=None,
                     help="Override deck.time.steps (useful for smoke tests).")
    run.add_argument("--log-every", type=int, default=0,
                     help="Print progress + record scalar diagnostics every N steps (0 = off).")
    run.add_argument("--write-every", type=int, default=0,
                     help="Write a self-contained per-step snapshot to <out>_<step>.npz every N steps (0 = end-of-run out.npz only).")
    run.set_defaults(func=_do_run)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
