"""``quasar pic`` command-line entry point.

Subcommands:

* ``run <input.yaml>`` — build the solver, seed species, step, dump ``out.npz``.

Run with::

    python -m quasar.pic.cli run examples/square_toroid_pic/input.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import _core
from .._paths import confine_output_path
from ..coil.io import build_conductor_system
from . import initial_conditions as ic
from . import io as pic_io
from ._units import QE as EV_TO_J  # elementary charge in C == eV->J factor
from ._units import Units
from .numerics import (besselj0, cfl_dt, cfl_limit, cyl_cfl_dt, cyl_cfl_limit,
                       j0_zero)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


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


def _cyl_cfl_dt_internal(domain, units: Units) -> float:
    dr, dz = _internal_spacing(domain, units)
    return cyl_cfl_dt(dr, dz, c=1.0)


def _cyl_cfl_limit_internal(domain, units: Units) -> float:
    dr, dz = _internal_spacing(domain, units)
    return cyl_cfl_limit(dr, dz, c=1.0)


def _make_solver(deck: pic_io.PicDeck, units: Units):
    pic = _core.pic
    # Grid coordinates enter the solver in internal length units. The ghost halo
    # must be wide enough for the FDTD order (order 4 reads two cells past a wall).
    grid = pic.Grid2D(nx=deck.domain.nx, ny=deck.domain.ny,
                      lx=units.length(deck.domain.lx_m),
                      ly=units.length(deck.domain.ly_m),
                      origin_x=units.length(deck.domain.origin_x_m),
                      origin_y=units.length(deck.domain.origin_y_m),
                      nghost=pic.required_nghost(deck.numerics.fdtd_order))
    cfg = pic.EmPicConfig()
    cfg.grid = grid
    cfg.fdtd_order = deck.numerics.fdtd_order
    cfg.shape = deck.numerics.shape
    cfg.plane = deck.plane
    cfg.geometry = deck.geometry
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
        # Thermal speed is computed in SI from SI mass/temperature, then mapped to
        # the internal velocity unit (c). Positions/areas/density are converted to
        # internal units so the macro-particle weight (density*area/N) is internal.
        v_thermal_si = float(np.sqrt(sp.initial.temperature_eV * EV_TO_J / sp.mass_kg))
        v_thermal = units.velocity(v_thermal_si)
        drift = tuple(units.velocity(c) for c in sp.initial.drift_v)
        if sp.initial.distribution == "maxwellian_block":
            x_min = units.length(sp.initial.region_x_min_m)
            x_max = units.length(sp.initial.region_x_max_m)
            y_min = units.length(sp.initial.region_y_min_m)
            y_max = units.length(sp.initial.region_y_max_m)
            positions = ic.quiet_positions_2d_block(
                sp.n_particles, x_min, x_max, y_min, y_max)
            cell_area = ic.quiet_block_cell_area(
                sp.n_particles, x_min, x_max, y_min, y_max)
        else:
            lx = units.length(deck.domain.lx_m)
            ly = units.length(deck.domain.ly_m)
            positions = ic.quiet_positions_2d(sp.n_particles, lx, ly)
            positions = positions + np.array([units.length(deck.domain.origin_x_m),
                                              units.length(deck.domain.origin_y_m)])
            cell_area = ic.quiet_block_cell_area(sp.n_particles, 0.0, lx, 0.0, ly)
        velocities = ic.maxwellian(sp.n_particles, v_thermal,
                                   drift=drift,
                                   seed=int(rng.integers(0, 2**31 - 1)))

        # Each quiet-start particle represents one layout cell, so the macro-weight
        # is density * cell_area. This keeps the local number density exactly
        # `density` even when n_particles is not a perfect square (the truncated
        # last row would otherwise bias density by side**2 / n_particles).
        macro_weight = units.density(sp.initial.density_per_m3) * cell_area
        weights = np.full(sp.n_particles, macro_weight)

        cfg = pic.SpeciesConfig(name=sp.name,
                                charge=units.charge(sp.charge_C),
                                mass=units.mass(sp.mass_kg),
                                capacity=sp.n_particles)
        idx = solver.add_species(cfg)
        # The pybind binding is declared with py::array::forcecast, so it copies
        # to contiguous float64 itself; no host-side astype needed here.
        solver.species_at(idx).set_host_particles(
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
    # reads the keys it knows (e.g. uniform reads b0/e0); unknown keys are ignored,
    # so a parameterless evaluator like biot_savart simply gets an empty configure.
    evaluator = ms.create_field_evaluator(ef.evaluator_type)
    evaluator.configure(ef.evaluator_params())
    cs = build_conductor_system(ef.conductors)
    length_scale, e_field_scale, b_field_scale = units.external_scales()
    solver.sample_external_field(
        evaluator, cs, length_scale, e_field_scale, b_field_scale,
        plane=deck.plane)


def _seed_fields(solver, deck: pic_io.PicDeck) -> None:
    """Apply an optional initial field seed (normalized-unit decks only).

    The interior is written on a ghost-padded buffer matching the solver storage
    (pitch = nx + 2*nghost). Amplitudes are dimensionless / internal units."""
    init = deck.fields.initial
    if init is None:
        return
    nx, ny = deck.domain.nx, deck.domain.ny
    # The deck carries the FDTD order, so the ghost width is known directly
    # (required_nghost is the C++ authority); no need to reverse-engineer it from
    # the storage size.
    g = _core.pic.required_nghost(deck.numerics.fdtd_order)
    pitch = nx + 2 * g
    height = ny + 2 * g

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))  # (ny, nx) index grids

    def _seed(component: str, interior: np.ndarray) -> None:
        # Write the vectorized interior into the ghost-padded buffer's interior.
        buf = np.zeros((height, pitch), dtype=np.float64)
        buf[g:g + ny, g:g + nx] = interior
        solver.seed_field(component, buf.reshape(-1))

    comp = init.component.lower()
    if init.type == "seed_perturbation":
        mx = max(1, init.mode[0])
        amp = init.amplitude
        if deck.geometry == "cylindrical":
            # On a cylindrical (r,z) grid the radial axis is not a plain Cartesian
            # interval: a sin(2 pi r/R) perturbation is dominated by higher radial
            # Bessel modes (it has ~90% overlap with TM020, only ~3% with TM010),
            # so the cavity would ring at the wrong line. Seed the physically
            # appropriate axisymmetric radial eigenmode J0(j_{0,mx} r/R) instead,
            # which excites the TM0,mx,0 mode cleanly (mx=1 -> TM010). Uniform in z.
            # j0_zero / besselj0 are dependency-free (numpy only) so a cylindrical
            # deck does not pull in scipy.
            j0n = j0_zero(mx)
            r_over_R = (ii + 0.5) / nx  # r/R for cell-centred radius, R = lx
            _seed(comp, amp * besselj0(j0n * r_over_R))
        else:
            _seed(comp, amp * np.sin(2 * np.pi * mx * (ii + 0.5) / nx))
    elif init.type == "seed_em_wave":
        mx, my = init.mode
        amp = init.amplitude
        # Only +x propagation is implemented; a non-zero my would request a +y /
        # oblique wave that this seeding does not build, so reject it loudly rather
        # than silently seeding a +x wave instead.
        if my != 0:
            raise ValueError(
                f"seed_em_wave: only +x propagation (mode my=0) is supported, "
                f"got mode={init.mode!r}")
        # A +x-propagating wave: Ez = sin(kx), By = -Ez (c = 1 internal units).
        wave = amp * np.sin(2 * np.pi * mx * (ii + 0.5) / nx)
        if comp == "ez":
            _seed("ez", wave)
            _seed("by", -wave)
        elif comp == "ey":
            _seed("ey", wave)
            _seed("bz", wave)
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
        "nghost": _core.pic.required_nghost(deck.numerics.fdtd_order),
        # Which lab plane the grid represents, so offline readers can label the
        # in-plane axes and know that external_b{x,y,z} are PIC-frame components
        # (in "xz" mode the out-of-plane B_phi lands in external_bz).
        "plane": deck.plane,
        # Coordinate system of the grid so offline readers know whether to treat
        # the in-plane axes as (x,y) or axisymmetric r-z.
        "geometry": deck.geometry,
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
        "external_bx": final["external_bx"],
        "external_by": final["external_by"],
        "external_bz": final["external_bz"],
    }
    for name, arr in final["fields"].items():
        flat[f"field_{name}"] = arr
    if "species" in final:
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


def prepare_run(deck: pic_io.PicDeck, units: Units, *, seed: int = 0):
    """Build a fully seeded solver from a parsed deck, ready to ``step(dt)``.

    Constructs the solver, applies the external field and any field seed, seeds
    all species, and resolves the internal/SI timestep. Returns
    ``(solver, species_indices, dt, dt_si)``. This is the shared build path for
    the ``run`` subcommand and for offline drivers (e.g. example plot scripts)
    so the solver-construction + seeding logic lives in one place."""
    solver = _make_solver(deck, units)
    _apply_external_field(solver, deck, units)
    _seed_fields(solver, deck)
    rng = np.random.default_rng(seed)
    species_indices = _seed_species(solver, deck, units, rng)

    # The solver steps in internal time units; an explicit deck dt_s is SI and is
    # converted, while "auto" is derived directly on the internal grid (c = 1).
    cylindrical = deck.geometry == "cylindrical"
    if deck.time.dt_s == "auto":
        if cylindrical:
            dt = _cyl_cfl_dt_internal(deck.domain, units)
        else:
            dt = _cfl_dt_internal(deck.domain, units, deck.numerics.fdtd_order)
    else:
        dt = units.time(float(deck.time.dt_s))
        # An explicit dt above the Yee CFL limit makes the FDTD update diverge
        # exponentially; the 'auto' path is CFL-safe by construction but a
        # user-supplied dt must be checked against the same limit.
        if cylindrical:
            cfl = _cyl_cfl_limit_internal(deck.domain, units)
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
    dt_si = units.time_to_si(dt)
    return solver, species_indices, dt, dt_si


def _do_run(args: argparse.Namespace) -> int:
    deck_path = Path(args.input).resolve()
    deck = pic_io.load(deck_path)
    if args.steps_override is not None:
        deck.time = pic_io.Time(dt_s=deck.time.dt_s, steps=args.steps_override)
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
        print(f"dt     : {dt_si:.6e} s    steps: {deck.time.steps}")

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
        np.savez(out_path, **_flatten_for_npz(snapshots, final, series))

    log_every = max(0, int(args.log_every))
    write_every = max(0, int(args.write_every))

    # sim_time accumulates SI seconds (dt_si) so output time_s is physical, while
    # the solver is advanced by the internal-unit dt.
    for step in range(deck.time.steps):
        solver.step(dt)
        sim_time += dt_si
        step_done = step + 1
        if deck.diagnostics.cadence > 0 and step_done % deck.diagnostics.cadence == 0:
            snapshots.append(_snapshot(solver, deck, species_indices,
                                       step_done, sim_time, units))
        if log_every > 0 and step_done % log_every == 0:
            _record_scalars(step_done, sim_time)
            elapsed = time.time() - t0
            rate = step_done / elapsed if elapsed > 0 else 0.0
            remaining = (deck.time.steps - step_done) / rate if rate > 0 else float("nan")
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
            np.savez(per_path, **_flatten_for_npz([], final, None))

    # The deposit defers its overflow check off the per-step hot path, so drain
    # the accumulated flag once after the last step (raises "reduce dt" if any
    # deposit spilled the deposition window).
    solver.finalize()

    if log_every == 0 or deck.time.steps % log_every != 0:
        _record_scalars(deck.time.steps, sim_time)
    _flush(deck.time.steps, sim_time)


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
