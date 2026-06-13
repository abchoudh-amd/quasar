"""``quasar mhd`` command-line entry point.

Subcommands:

* ``run <input.yaml>`` — build the solver, seed the initial condition, step,
  dump ``out.npz``.

Run with::

    python -m quasar.mhd.cli run examples/orszag_tang/input.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import _core
from .._paths import confine_output_path, positive_int as _positive_int
from . import io as mhd_io


def _make_config(deck: mhd_io.MhdDeck):
    mhd = _core.mhd
    # Pass nghost=0 so the solver ctor sizes the working grid's halo from the
    # reconstruction scheme's required_nghost() (a positive deck nghost smaller
    # than that required halo is a hard error). The solver's actual grid().nghost
    # is read back after construction to lay out the seeded initial condition.
    grid = mhd.Grid2D(nx=deck.domain.nx, ny=deck.domain.ny,
                      lx=deck.domain.lx_m, ly=deck.domain.ly_m,
                      origin_x=deck.domain.origin_x_m,
                      origin_y=deck.domain.origin_y_m,
                      nghost=0)
    cfg = mhd.MhdConfig()
    cfg.grid = grid
    cfg.gamma = deck.numerics.gamma
    cfg.geometry = deck.geometry
    cfg.reconstruction = deck.numerics.reconstruction
    cfg.riemann = deck.numerics.riemann
    cfg.integrator = deck.numerics.integrator
    cfg.ct = deck.numerics.ct
    cfg.positivity = deck.numerics.positivity
    cfg.rho_floor = deck.numerics.rho_floor
    cfg.p_floor = deck.numerics.p_floor
    cfg.cfl = deck.numerics.cfl
    # Boundaries are selected by registry name; the deck strings (already
    # validated in io.py) pass straight through to the C++ registry.
    for side, name in enumerate(deck.boundary.fluid):
        cfg.boundary.set_fluid_side(side, name)
    for side, name in enumerate(deck.boundary.field):
        cfg.boundary.set_field_side(side, name)
    # Static background field B0 (field-split B = B0 + b). When disabled the
    # solver runs its zero-B0 fast path; the uniform-vector params are still
    # carried through but only consumed by the "uniform" profile. Mutate the
    # config's default-constructed MhdBackgroundSpec in place, mirroring how
    # cfg.boundary is populated above.
    cfg.background.enabled = deck.background.enabled
    cfg.background.profile = deck.background.profile
    cfg.background.bx0 = deck.background.bx0
    cfg.background.by0 = deck.background.by0
    cfg.background.bz0 = deck.background.bz0
    return cfg


def prepare_run(deck: mhd_io.MhdDeck):
    """Build a fully seeded solver from a parsed deck, ready to ``step(dt)``.

    Constructs the solver (which fixes the ghost width from the reconstruction
    scheme), builds the initial condition on that ghost-padded layout, seeds
    every conserved component, and resolves the timestep with the CFL guard.
    Returns ``(solver, dt, dt_is_auto)``; ``dt`` is the t=0 timestep and
    ``dt_is_auto`` tells the run loop whether to recompute it each step.
    """
    cfg = _make_config(deck)
    solver = _core.mhd.MhdSolver2D(cfg)

    # The ctor may have widened nghost; seed on the solver's actual grid layout.
    nghost = solver.grid().nghost
    state = mhd_io.build_initial_state(deck, nghost)
    for component, buf in state.items():
        solver.seed_state(component, buf)

    # Static background field B0 (field-split B = B0 + b). Seed it BEFORE the
    # cfl_limit() probe so the CFL guard sees the total field B0 + b: the fast
    # magnetosonic speed uses the total field, so a guide field B0 tightens the
    # stable dt relative to the B0 = 0 case. build_background_field returns None
    # when the background is disabled (zero-B0 fast path), in which case nothing
    # is seeded.
    background = mhd_io.build_background_field(deck, nghost)
    if background is not None:
        for component in ("b0x", "b0y", "b0z"):
            solver.seed_background(component, background[component])

    # cfl_limit() scans the live state for the max fast-magnetosonic signal speed
    # (and already folds in the cfl safety factor), so it is only meaningful AFTER
    # seeding and it TIGHTENS as the flow develops. For an explicit deck dt we
    # reject an over-limit value here, before any stepping (mirrors the PIC CFL
    # guard); for "auto" the run loop recomputes the safe dt each step.
    cfl = solver.cfl_limit()
    if deck.time.dt_s == "auto":
        dt = cfl
        dt_is_auto = True
    else:
        dt = float(deck.time.dt_s)
        dt_is_auto = False
        if dt > cfl:
            raise ValueError(
                f"time.dt_s ({dt:.6e}) exceeds the MHD CFL stability limit "
                f"({cfl:.6e}) for this grid and seeded state; reduce dt_s or use "
                f"'auto'.")
    return solver, dt, dt_is_auto


def _interior_slice(arr2d: np.ndarray, nx: int, ny: int, nghost: int) -> np.ndarray:
    g = nghost
    return arr2d[g:g + ny, g:g + nx]


def _component_2d(solver, deck: mhd_io.MhdDeck, component: str) -> np.ndarray:
    """State component as a (height, pitch) ghost-padded 2-D array."""
    nghost = solver.grid().nghost
    pitch = deck.domain.nx + 2 * nghost
    height = deck.domain.ny + 2 * nghost
    flat = np.asarray(solver.state_component_to_host(component))
    return flat.reshape(height, pitch)


def _orszag_tang_invariants(solver, deck: mhd_io.MhdDeck) -> dict:
    """Sum of rho and energy over the interior at the current time.

    Emitted as mass_initial / energy_initial for the orszag_tang token (a sibling
    example/test compares the final conserved totals to these).
    """
    nx, ny, nghost = deck.domain.nx, deck.domain.ny, solver.grid().nghost
    rho = _interior_slice(_component_2d(solver, deck, "rho"), nx, ny, nghost)
    energy = _interior_slice(_component_2d(solver, deck, "energy"), nx, ny, nghost)
    return {"mass_initial": float(rho.sum()),
            "energy_initial": float(energy.sum())}


def _do_run(args: argparse.Namespace) -> int:
    deck_path = Path(args.input).resolve()
    deck = mhd_io.load(deck_path)
    if args.steps_override is not None:
        deck.time = mhd_io.Time(dt_s=deck.time.dt_s, steps=args.steps_override,
                                t_end=deck.time.t_end)
        deck.validate()

    solver, dt, dt_is_auto = prepare_run(deck)

    if args.print_config:
        print(f"deck    : {deck_path}")
        print(f"grid    : {deck.domain.nx}x{deck.domain.ny}  "
              f"({deck.domain.lx_m}x{deck.domain.ly_m})  "
              f"origin=({deck.domain.origin_x_m}, {deck.domain.origin_y_m})  "
              f"nghost={solver.grid().nghost}")
        print(f"geometry: {deck.geometry}")
        print(f"gamma   : {deck.numerics.gamma}")
        print(f"schemes : recon={deck.numerics.reconstruction} "
              f"riemann={deck.numerics.riemann} "
              f"integ={deck.numerics.integrator} ct={deck.numerics.ct} "
              f"pos={deck.numerics.positivity}")
        print(f"initial : {deck.initial.type}")
        print(f"dt      : {dt:.6e}    steps: {deck.time.steps}")

    out_path = confine_output_path(deck_path.parent, deck.diagnostics.output_path,
                                   label="diagnostics.output_path")
    _run_loop(solver, deck, dt, dt_is_auto, out_path, args)
    if args.print_config or args.verbose:
        print(f"wrote   : {out_path}")
    return 0


def _snapshot_fields(solver, deck: mhd_io.MhdDeck) -> dict:
    # Interior (ny, nx) cell-centered fields, matching the final state_* layout
    # written by _flatten_for_npz (the ghost halo is stripped here too).
    nx, ny, nghost = deck.domain.nx, deck.domain.ny, solver.grid().nghost
    return {name: _interior_slice(_component_2d(solver, deck, name), nx, ny, nghost)
            for name in deck.diagnostics.fields}


def _initial_state_2d(solver, deck: mhd_io.MhdDeck) -> dict:
    """Interior (ny, nx) cell-centered snapshot of every conserved component at
    the current time, keyed ``state_<name>_initial``. Captured at t=0 so a
    smooth-wave example can compare the final field against the exact (== seeded)
    analytic reference instead of a self-referential harmonic fit."""
    nx, ny, nghost = deck.domain.nx, deck.domain.ny, solver.grid().nghost
    out: dict[str, np.ndarray] = {}
    for name in mhd_io.STATE_COMPONENTS:
        padded = _component_2d(solver, deck, name)
        out[f"state_{name}_initial"] = _interior_slice(padded, nx, ny, nghost)
    return out


def _flatten_for_npz(solver, deck: mhd_io.MhdDeck, final_step: int,
                     final_time: float, divb_series: list,
                     snapshots: list[dict], extra_scalars: dict) -> dict:
    nghost = solver.grid().nghost
    flat: dict[str, np.ndarray] = {
        "final_step": np.array([final_step]),
        "final_time_s": np.array([final_time]),
        "nx": np.array([deck.domain.nx]),
        "ny": np.array([deck.domain.ny]),
        "nghost": np.array([nghost]),
        "geometry": np.array([deck.geometry]),
        "gamma": np.array([deck.numerics.gamma]),
    }
    # Final conserved state: full set of components (cell-centered B sampled),
    # written as the INTERIOR (nx*ny) field, row-major (ny, nx). The solver hands
    # back the full ghost-padded storage ((ny+2g)*(nx+2g)); the .npz contract
    # (and every state_* reader in the tests/post-processing) is the cell-centered
    # interior only, so strip the ghost halo here rather than leak the padding.
    nx, ny = deck.domain.nx, deck.domain.ny
    for name in mhd_io.STATE_COMPONENTS:
        padded = _component_2d(solver, deck, name)
        flat[f"state_{name}"] = _interior_slice(padded, nx, ny, nghost)
    # div B diagnostic: per-snapshot series plus the final scalar.
    flat["divb_linf"] = np.asarray(divb_series, dtype=np.float64)
    flat["divb_linf_final"] = np.array(
        [divb_series[-1] if divb_series else solver.divergence_b_max()])
    if snapshots:
        flat["snapshot_steps"] = np.array([s["step"] for s in snapshots])
        flat["snapshot_times_s"] = np.array([s["time_s"] for s in snapshots])
        for name in deck.diagnostics.fields:
            flat[f"snapshot_state_{name}"] = np.stack(
                [s["fields"][name] for s in snapshots])
        flat["snapshot_divb_linf"] = np.array(
            [s["divb"] for s in snapshots], dtype=np.float64)
    for k, v in extra_scalars.items():
        flat[k] = np.array([v])
    return flat


def _run_loop(solver, deck: mhd_io.MhdDeck, dt: float, dt_is_auto: bool, out_path,
              args: argparse.Namespace) -> None:
    snapshots: list[dict] = []
    divb_series: list[float] = []
    sim_time = 0.0
    t0 = time.time()

    # Orszag-Tang conserved-total invariants captured at t=0 (before stepping).
    extra_scalars: dict[str, float] = {}
    if deck.initial.type == "orszag_tang":
        extra_scalars.update(_orszag_tang_invariants(solver, deck))

    # Seeded (t=0) conserved state. For an exactly-periodic smooth wave the
    # analytic solution at an integer-wavelength output time equals this seed, so
    # convergence tests compare the final field against these state_*_initial keys.
    initial_state = _initial_state_2d(solver, deck)

    log_every = max(0, int(args.log_every))

    # Record the t=0 div B as the first series sample so the diagnostic captures
    # the seeded (machine-epsilon) value too.
    divb_series.append(float(solver.divergence_b_max()))

    cadence = deck.diagnostics.cadence
    for step in range(deck.time.steps):
        step_done = step + 1
        is_last = step_done == deck.time.steps
        # For "auto" dt the stable step tightens as the flow develops, so refresh
        # it from the live state before each step; an explicit deck dt is held
        # fixed (the solver rejects it if it later exceeds the limit). The auto
        # path uses step_unchecked: cfl_limit() was just computed, so step()'s
        # internal CFL re-reduction would be redundant.
        if dt_is_auto:
            dt = solver.cfl_limit()
            solver.step_unchecked(dt)
        else:
            solver.step(dt)
        sim_time += dt
        # divergence_b_max() runs a ghost refill + a syncing device reduction, so
        # sample it only on cadence steps, the final step (for divb_linf_final),
        # and when a progress log needs it -- not every step.
        on_cadence = cadence > 0 and step_done % cadence == 0
        need_divb = on_cadence or is_last or (
            log_every > 0 and step_done % log_every == 0)
        divb_now = float(solver.divergence_b_max()) if need_divb else None
        if divb_now is not None:
            divb_series.append(divb_now)
        if on_cadence:
            snapshots.append({
                "step": step_done,
                "time_s": sim_time,
                "fields": _snapshot_fields(solver, deck),
                "divb": divb_now,
            })
        if log_every > 0 and step_done % log_every == 0:
            elapsed = time.time() - t0
            rate = step_done / elapsed if elapsed > 0 else 0.0
            remaining = (deck.time.steps - step_done) / rate if rate > 0 else float("nan")
            print(f"step {step_done}/{deck.time.steps}  "
                  f"t={sim_time:.6e}  rate={rate:.0f} step/s  "
                  f"eta={remaining:.0f}s  |divB|inf={divb_now:.3e}",
                  flush=True)

    flat = _flatten_for_npz(
        solver, deck, deck.time.steps, sim_time, divb_series, snapshots,
        extra_scalars)
    flat.update(initial_state)
    np.savez(out_path, **flat)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quasar-mhd",
                                description="Quasar ideal-MHD driver.")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run an MHD simulation from a YAML deck.")
    run.add_argument("input", help="Path to the YAML deck.")
    run.add_argument("--verbose", action="store_true",
                     help="print informational output (default: quiet)")
    run.add_argument("--print-config", action="store_true",
                     help="Print resolved deck + dt before running.")
    run.add_argument("--steps-override", type=_positive_int, default=None,
                     help="Override deck.time.steps (useful for smoke tests).")
    run.add_argument("--log-every", type=int, default=0,
                     help="Print progress every N steps (0 = off).")
    run.set_defaults(func=_do_run)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
