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
from . import initial_conditions as ic
from . import io as pic_io
from ._units import Units
from .numerics import cfl_dt

EV_TO_J = 1.602176634e-19


def _cfl_dt_internal(domain, units: Units, fdtd_order: int = 2) -> float:
    # The solver runs in internal units where c = 1, so the CFL is evaluated on
    # the internal grid spacing with c = 1 (for a normalized deck the lengths are
    # already internal and this is the identity).
    dx = units.length(domain.lx_m) / domain.nx
    dy = units.length(domain.ly_m) / domain.ny
    return cfl_dt(dx, dy, c=1.0, fdtd_order=fdtd_order)


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
    cfg.shape_order = 2 if deck.numerics.shape == "tsc" else 1
    if units.normalization is not None:
        cfg.normalization = units.normalization
    kind_map = {
        "periodic": pic.ParticleBoundaryKind.periodic,
        "specular": pic.ParticleBoundaryKind.specular,
        "absorbing": pic.ParticleBoundaryKind.absorbing,
    }
    for side, name in enumerate(deck.boundary.particle):
        cfg.boundary.set_particle_side(side, kind_map[name])
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
            occupied_area = (x_max - x_min) * (y_max - y_min)
        else:
            lx = units.length(deck.domain.lx_m)
            ly = units.length(deck.domain.ly_m)
            positions = ic.quiet_positions_2d(sp.n_particles, lx, ly)
            positions = positions + np.array([units.length(deck.domain.origin_x_m),
                                              units.length(deck.domain.origin_y_m)])
            occupied_area = lx * ly
        velocities = ic.maxwellian(sp.n_particles, v_thermal,
                                   drift=drift,
                                   seed=int(rng.integers(0, 2**31 - 1)))

        macro_weight = (units.density(sp.initial.density_per_m3) * occupied_area
                        / sp.n_particles)
        weights = np.full(sp.n_particles, macro_weight)

        cfg = pic.SpeciesConfig(name=sp.name,
                                charge=units.charge(sp.charge_C),
                                mass=units.mass(sp.mass_kg),
                                capacity=sp.n_particles)
        idx = solver.add_species(cfg)
        solver.species_at(idx).set_host_particles(
            x=positions[:, 0].astype(np.float64),
            y=positions[:, 1].astype(np.float64),
            vx=velocities[:, 0].astype(np.float64),
            vy=velocities[:, 1].astype(np.float64),
            vz=velocities[:, 2].astype(np.float64),
            weight=weights.astype(np.float64),
        )
        indices.append(idx)
    return indices


def _apply_external_field(solver, deck: pic_io.PicDeck, units: Units) -> None:
    if deck.external_field is None:
        return
    if deck.external_field.evaluator_type != "biot_savart":
        raise NotImplementedError(
            f"evaluator {deck.external_field.evaluator_type!r} is not bound yet")
    cs = pic_io.build_conductor_system(deck.external_field.conductors)
    evaluator = _core.magnetostatics.BiotSavartEvaluator()
    length_scale, e_field_scale, b_field_scale = units.external_scales()
    solver.sample_external_field_biot_savart(
        evaluator, cs, length_scale, e_field_scale, b_field_scale)


def _field_to_si(name: str, arr: np.ndarray, units: Units) -> np.ndarray:
    # E components (ex/ey/ez) and B components (bx/by/bz) carry different unit
    # scales; output is reported in SI so downstream consumers see physical values.
    if name in ("ex", "ey", "ez"):
        return units.e_field_to_si(arr)
    return units.b_field_to_si(arr)


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
    fields_d = solver.fields_to_host()
    external_d = solver.external_fields_to_host()
    snap = {
        "step": step,
        "time_s": sim_time,
        "fields": {k: _field_to_si(k, fields_d[k], units)
                   for k in deck.diagnostics.fields if k in fields_d},
        "external_bx": units.b_field_to_si(external_d["bx"]),
        "external_by": units.b_field_to_si(external_d["by"]),
        "external_bz": units.b_field_to_si(external_d["bz"]),
        "nx": fields_d["nx"],
        "ny": fields_d["ny"],
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


def _do_run(args: argparse.Namespace) -> int:
    deck_path = Path(args.input).resolve()
    deck = pic_io.load(deck_path)
    if args.steps_override is not None:
        deck.time = pic_io.Time(dt_s=deck.time.dt_s, steps=args.steps_override)

    units = Units(deck)
    solver = _make_solver(deck, units)
    _apply_external_field(solver, deck, units)
    rng = np.random.default_rng(args.seed)
    species_indices = _seed_species(solver, deck, units, rng)

    # The solver steps in internal time units; an explicit deck dt_s is SI and is
    # converted, while "auto" is derived directly on the internal grid (c = 1).
    if deck.time.dt_s == "auto":
        dt = _cfl_dt_internal(deck.domain, units, deck.numerics.fdtd_order)
    else:
        dt = units.time(float(deck.time.dt_s))
    dt_si = units.time_to_si(dt)
    if args.print_config:
        print(f"deck   : {deck_path}")
        print(f"grid   : {deck.domain.nx}x{deck.domain.ny}  "
              f"({deck.domain.lx_m}x{deck.domain.ly_m}) m  "
              f"origin=({deck.domain.origin_x_m}, {deck.domain.origin_y_m})")
        print(f"units  : {deck.units}")
        print(f"species: {[sp.name for sp in deck.species]}")
        print(f"dt     : {dt_si:.6e} s    steps: {deck.time.steps}")

    snapshots: list[dict] = []
    sim_time = 0.0
    # Confine the deck-supplied output path to the deck's own directory so a
    # stray absolute path or "../" cannot write outside it.
    base = deck_path.parent.resolve()
    out_path = (base / deck.diagnostics.output_path).resolve()
    if not out_path.is_relative_to(base):
        raise ValueError(
            f"diagnostics.output_path {deck.diagnostics.output_path!r} escapes "
            f"the deck directory {base}")
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

    def _checkpoint(step_done: int, t_now: float) -> None:
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
            _checkpoint(step_done, sim_time)

    if log_every == 0 or deck.time.steps % log_every != 0:
        _record_scalars(deck.time.steps, sim_time)
    final = _snapshot(solver, deck, species_indices, deck.time.steps, sim_time, units)
    np.savez(out_path, **_flatten_for_npz(snapshots, final, series))
    if args.print_config:
        print(f"wrote  : {out_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quasar.pic.cli",
                                description="Quasar PIC driver.")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a PIC simulation from a YAML deck.")
    run.add_argument("input", help="Path to the YAML deck.")
    run.add_argument("--seed", type=int, default=0,
                     help="RNG seed for initial-condition sampling.")
    run.add_argument("--print-config", action="store_true",
                     help="Print resolved deck + dt before running.")
    run.add_argument("--steps-override", type=int, default=None,
                     help="Override deck.time.steps (useful for smoke tests).")
    run.add_argument("--log-every", type=int, default=0,
                     help="Print progress + record scalar diagnostics every N steps (0 = off).")
    run.add_argument("--write-every", type=int, default=0,
                     help="Flush rolling checkpoint to out.npz every N steps (0 = end-of-run only).")
    run.set_defaults(func=_do_run)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
