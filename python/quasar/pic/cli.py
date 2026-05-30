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

EV_TO_J = 1.602176634e-19
C_LIGHT = 299792458.0


def _cfl_dt(domain) -> float:
    dx = domain.lx_m / domain.nx
    dy = domain.ly_m / domain.ny
    return 0.5 / (C_LIGHT * np.sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy)))


def _make_solver(deck: pic_io.PicDeck):
    pic = _core.pic
    grid = pic.Grid2D(nx=deck.domain.nx, ny=deck.domain.ny,
                      lx=deck.domain.lx_m, ly=deck.domain.ly_m,
                      origin_x=deck.domain.origin_x_m,
                      origin_y=deck.domain.origin_y_m)
    cfg = pic.EmPicConfig()
    cfg.grid = grid
    cfg.fdtd_order = deck.numerics.fdtd_order
    cfg.shape_order = 2 if deck.numerics.shape == "tsc" else 1
    kind_map = {
        "periodic": pic.ParticleBoundaryKind.periodic,
        "specular": pic.ParticleBoundaryKind.specular,
        "absorbing": pic.ParticleBoundaryKind.absorbing,
    }
    for side, name in enumerate(deck.boundary.particle):
        cfg.boundary.set_particle_side(side, kind_map[name])
    return pic.EmPic2D3V(cfg)


def _seed_species(solver, deck: pic_io.PicDeck,
                  rng: np.random.Generator) -> list[int]:
    pic = _core.pic
    indices: list[int] = []
    for sp in deck.species:
        v_thermal = float(np.sqrt(sp.initial.temperature_eV * EV_TO_J / sp.mass_kg))
        if sp.initial.distribution == "maxwellian_block":
            x_min = sp.initial.region_x_min_m
            x_max = sp.initial.region_x_max_m
            y_min = sp.initial.region_y_min_m
            y_max = sp.initial.region_y_max_m
            positions = ic.quiet_positions_2d_block(
                sp.n_particles, x_min, x_max, y_min, y_max)
            occupied_area = (x_max - x_min) * (y_max - y_min)
        else:
            positions = ic.quiet_positions_2d(sp.n_particles, deck.domain.lx_m,
                                              deck.domain.ly_m)
            positions = positions + np.array([deck.domain.origin_x_m,
                                              deck.domain.origin_y_m])
            occupied_area = deck.domain.lx_m * deck.domain.ly_m
        velocities = ic.maxwellian(sp.n_particles, v_thermal,
                                   drift=sp.initial.drift_v,
                                   seed=int(rng.integers(0, 2**31 - 1)))

        macro_weight = (sp.initial.density_per_m3 * occupied_area
                        / sp.n_particles)
        weights = np.full(sp.n_particles, macro_weight)

        cfg = pic.SpeciesConfig(name=sp.name, charge=sp.charge_C,
                                mass=sp.mass_kg, capacity=sp.n_particles)
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


def _apply_external_field(solver, deck: pic_io.PicDeck) -> None:
    if deck.external_field is None:
        return
    if deck.external_field.evaluator_type != "biot_savart":
        raise NotImplementedError(
            f"evaluator {deck.external_field.evaluator_type!r} is not bound yet")
    cs = pic_io.build_conductor_system(deck.external_field.conductors)
    evaluator = _core.magnetostatics.BiotSavartEvaluator()
    solver.sample_external_field_biot_savart(evaluator, cs)


def _snapshot(solver, deck: pic_io.PicDeck, species_indices: list[int],
              step: int, sim_time: float) -> dict:
    fields_d = solver.fields_to_host()
    external_d = solver.external_fields_to_host()
    snap = {
        "step": step,
        "time_s": sim_time,
        "fields": {k: fields_d[k] for k in deck.diagnostics.fields if k in fields_d},
        "external_bx": external_d["bx"],
        "external_by": external_d["by"],
        "external_bz": external_d["bz"],
        "nx": fields_d["nx"],
        "ny": fields_d["ny"],
    }
    if deck.diagnostics.per_species:
        per_sp = {}
        for idx, sp in zip(species_indices, deck.species):
            per_sp[sp.name] = solver.species_at(idx).to_host()
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

    solver = _make_solver(deck)
    _apply_external_field(solver, deck)
    rng = np.random.default_rng(args.seed)
    species_indices = _seed_species(solver, deck, rng)

    dt = _cfl_dt(deck.domain) if deck.time.dt_s == "auto" else float(deck.time.dt_s)
    if args.print_config:
        print(f"deck   : {deck_path}")
        print(f"grid   : {deck.domain.nx}x{deck.domain.ny}  "
              f"({deck.domain.lx_m}x{deck.domain.ly_m}) m  "
              f"origin=({deck.domain.origin_x_m}, {deck.domain.origin_y_m})")
        print(f"species: {[sp.name for sp in deck.species]}")
        print(f"dt     : {dt:.6e} s    steps: {deck.time.steps}")

    snapshots: list[dict] = []
    sim_time = 0.0
    out_path = (deck_path.parent / deck.diagnostics.output_path).resolve()
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
        final = _snapshot(solver, deck, species_indices, step_done, t_now)
        np.savez(out_path, **_flatten_for_npz(snapshots, final, series))

    log_every = max(0, int(args.log_every))
    write_every = max(0, int(args.write_every))

    for step in range(deck.time.steps):
        solver.step(dt)
        sim_time += dt
        step_done = step + 1
        if deck.diagnostics.cadence > 0 and step_done % deck.diagnostics.cadence == 0:
            snapshots.append(_snapshot(solver, deck, species_indices,
                                       step_done, sim_time))
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
    final = _snapshot(solver, deck, species_indices, deck.time.steps, sim_time)
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
