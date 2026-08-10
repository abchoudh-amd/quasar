"""``quasar mhd`` command-line entry point.

Subcommands:

* ``run <input.yaml>`` — build the solver, seed the initial condition, step,
  dump ``out.npz``.

Run with::

    python -m quasar.mhd.cli run examples/orszag_tang/input.yaml
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import _core
from .. import distributed as _distributed
from .._paths import confine_output_path, positive_int as _positive_int
from . import io as mhd_io
from . import _units as mhd_units


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
    if not deck.background.enabled:
        return cfg
    analytic_profile = (
        deck.background.file is None and deck.background.a_file is None
        and deck.background.conductors is None
    )
    # Explicit modes need the native buffers allocated, but their deck profile
    # is intentionally irrelevant. Construct a harmless zero placeholder until
    # prepare_run replaces all three components; this avoids validating or
    # sampling an ignored analytic profile first.
    cfg.background.profile = (
        deck.background.profile if analytic_profile else "uniform"
    )
    cfg.background.bx0 = deck.background.bx0 if analytic_profile else 0.0
    cfg.background.by0 = deck.background.by0 if analytic_profile else 0.0
    cfg.background.bz0 = deck.background.bz0 if analytic_profile else 0.0
    # Keep analytic sampling inside the native solver so the registry profile's
    # trusted curl-free capability survives. A uniform output scale performs
    # the SI tesla -> mu0=1 conversion without replacing those samples.
    cfg.background.profile_scale = (
        float(mhd_units.magnetic_to_internal(1.0, deck.units))
        if analytic_profile else 1.0
    )
    # The opt-in annular vacuum solve is the only explicitly seeded construction
    # that carries a trusted domain-wide curl-free proof. Native validation is a
    # defense-in-depth contradiction check, not the source of that proof.
    cfg.background.curl_free = bool(
        deck.background.enabled
        and (deck.background.a_file is not None
             or deck.background.conductors is not None)
        and deck.background.params.get("vacuum_project", False)
        and deck.background.bz0 == 0.0
    )
    # Only analytic mode configures the native registry profile. File and
    # vector-potential modes override all three buffers immediately after
    # construction; their params (for example b_scale/vacuum_project) belong to
    # the Python loaders and are not analytic-profile parameters.
    if (deck.background.file is None and deck.background.a_file is None
            and deck.background.conductors is None):
        cfg.background.params = {
            key: float(value) for key, value in deck.background.params.items()
        }
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

    # Static background field B0 (field-split B = B0 + b). Analytic profiles
    # were already sampled by the native constructor; only explicit file or
    # vector-potential modes replace those buffers here. Do this BEFORE
    # cfl_limit() so the fast magnetosonic speed sees the final total field B0+b.
    explicit_background = (
        deck.background.enabled
        and (deck.background.file is not None
             or deck.background.a_file is not None
             or deck.background.conductors is not None)
    )
    if explicit_background:
        background = mhd_io.build_background_field(deck, nghost)
        assert background is not None
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
    if component in ("bx", "by", "bz"):
        flat = mhd_units.magnetic_to_output(flat, deck.units)
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


def _serial_run(input_deck: mhd_io.MhdDeck | str | Path, *,
                steps_override: int | None, verbose: bool, print_config: bool,
                log_every: int) -> _distributed.RunResult:
    if isinstance(input_deck, mhd_io.MhdDeck):
        deck_path: Path | None = None
        deck = input_deck
    else:
        deck_path = Path(input_deck).resolve()
        deck = mhd_io.load(deck_path)
    if steps_override is not None:
        # Keep run-local termination policy from mutating an in-memory deck
        # supplied by a Python caller.
        deck = copy.deepcopy(deck)
        deck.time = mhd_io.Time(dt_s=deck.time.dt_s, steps=steps_override,
                                t_end=deck.time.t_end)
        deck.validate()

    solver, dt, dt_is_auto = prepare_run(deck)

    if print_config:
        print(f"deck    : {deck_path if deck_path is not None else '<in-memory>'}")
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

    deck_directory = deck_path.parent if deck_path is not None else Path.cwd()
    out_path = confine_output_path(deck_directory, deck.diagnostics.output_path,
                                   label="diagnostics.output_path")
    final_step, final_time = _run_loop(
        solver, deck, dt, dt_is_auto, out_path,
        argparse.Namespace(log_every=log_every))
    if print_config or verbose:
        print(f"wrote   : {out_path}")
    return _distributed.RunResult(
        final_step=final_step,
        final_time=final_time,
        diagnostics_path=out_path,
        distributed=False,
    )


def run(input_deck: mhd_io.MhdDeck | str | Path, *,
        options: _distributed.RunOptions | None = None,
        steps_override: int | None = None, verbose: bool = False,
        print_config: bool = False, log_every: int = 0,
        ) -> _distributed.RunResult:
    """Run an MHD deck through the serial or explicitly selected distributed path.

    Passing ``options`` is an explicit distributed request; unavailable builds
    raise rather than silently falling back to the serial solver.
    """

    if steps_override is not None:
        if (isinstance(steps_override, bool)
                or not isinstance(steps_override, int)
                or steps_override <= 0):
            raise ValueError("steps_override must be a positive integer")
    if options is not None:
        if not isinstance(options, _distributed.RunOptions):
            raise TypeError(
                "options must be a quasar.distributed.RunOptions instance")
        return _distributed._execute(
            "mhd", input_deck, options, steps_override=steps_override,
            verbose=verbose, print_config=print_config,
            log_every=log_every)
    return _serial_run(
        input_deck, steps_override=steps_override, verbose=verbose,
        print_config=print_config, log_every=log_every)


class _DistributedUsageError(ValueError):
    """Internal marker for distributed CLI cross-option validation errors."""


def _distributed_options_from_args(
        args: argparse.Namespace) -> _distributed.RunOptions | None:
    names = ("devices", "decomposition", "transport", "diagnostics_layout",
             "checkpoint", "checkpoint_every", "restart")
    if not any(getattr(args, name, None) is not None for name in names):
        return None
    try:
        return _distributed.RunOptions(
            devices=args.devices if args.devices is not None else "auto",
            decomposition=(args.decomposition
                           if args.decomposition is not None else "auto"),
            transport=args.transport if args.transport is not None else "auto",
            diagnostics_layout=(args.diagnostics_layout
                                if args.diagnostics_layout is not None
                                else "gathered"),
            checkpoint=args.checkpoint,
            checkpoint_every=args.checkpoint_every,
            restart=args.restart,
        )
    except (TypeError, ValueError) as exc:
        raise _DistributedUsageError(str(exc)) from exc


def _do_run(args: argparse.Namespace) -> int:
    run(
        args.input,
        options=_distributed_options_from_args(args),
        steps_override=args.steps_override,
        verbose=args.verbose,
        print_config=args.print_config,
        log_every=args.log_every,
    )
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
        "lx_m": np.array([deck.domain.lx_m]),
        "ly_m": np.array([deck.domain.ly_m]),
        "nghost": np.array([nghost]),
        "units": np.array([deck.units]),
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
    # div B is an opt-in diagnostic because each sample performs a synchronizing
    # device reduction. When disabled, emit no divergence keys and, critically,
    # do not sneak in a final reduction while serializing the archive.
    if deck.diagnostics.divb:
        divb_out = mhd_units.magnetic_to_output(
            np.asarray(divb_series, dtype=np.float64), deck.units)
        flat["divb_linf"] = divb_out
        flat["divb_linf_final"] = np.array(
            [divb_out[-1] if divb_series else mhd_units.magnetic_to_output(
                solver.divergence_b_max(), deck.units)])
    if snapshots:
        flat["snapshot_steps"] = np.array([s["step"] for s in snapshots])
        flat["snapshot_times_s"] = np.array([s["time_s"] for s in snapshots])
        for name in deck.diagnostics.fields:
            flat[f"snapshot_state_{name}"] = np.stack(
                [s["fields"][name] for s in snapshots])
        if deck.diagnostics.divb:
            flat["snapshot_divb_linf"] = mhd_units.magnetic_to_output(
                np.array([s["divb"] for s in snapshots], dtype=np.float64),
                deck.units)
    for k, v in extra_scalars.items():
        flat[k] = np.array([v])
    return flat


def _run_loop(solver, deck: mhd_io.MhdDeck, dt: float, dt_is_auto: bool, out_path,
              args: argparse.Namespace) -> tuple[int, float]:
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

    # Record t=0 only when requested. divergence_b_max() synchronizes the device,
    # so diagnostics.divb=false must avoid even this otherwise invisible cost.
    if deck.diagnostics.divb:
        divb_series.append(float(solver.divergence_b_max()))

    cadence = deck.diagnostics.cadence
    step_done = 0
    t_end = deck.time.t_end
    while step_done < deck.time.steps and (
            t_end is None or sim_time < float(t_end)):
        # For "auto" dt the stable step tightens as the flow develops, so refresh
        # it from the live state before each step; an explicit deck dt is held
        # fixed (the solver rejects it if it later exceeds the limit). The auto
        # path uses step_unchecked: cfl_limit() was just computed, so step()'s
        # internal CFL re-reduction would be redundant.
        dt_limit = float(solver.cfl_limit()) if dt_is_auto else float(dt)
        if not np.isfinite(dt_limit) or not dt_limit > 0.0:
            raise RuntimeError(
                "MHD timestep is not finite and strictly positive")
        dt_step = dt_limit
        clipped_to_end = False
        if t_end is not None:
            remaining = float(t_end) - sim_time
            dt_step = min(dt_step, remaining)
            clipped_to_end = dt_step == remaining
        if not np.isfinite(dt_step) or not dt_step > 0.0:
            raise RuntimeError(
                "MHD timestep cannot make positive finite progress")
        next_time = None
        if not clipped_to_end:
            next_time = sim_time + dt_step
            if not np.isfinite(next_time) or not next_time > sim_time:
                raise RuntimeError(
                    "MHD timestep is too small to advance simulation time")
        if dt_is_auto:
            solver.step_unchecked(dt_step)
        else:
            solver.step(dt_step)
        if clipped_to_end:
            # Assign the requested endpoint only after integrating the exact
            # residual interval selected above. No epsilon snap may report time
            # that was never advanced by the solver.
            sim_time = float(t_end)
        else:
            sim_time = next_time
        step_done += 1
        is_last = step_done == deck.time.steps or (
            t_end is not None and sim_time >= float(t_end))
        # divergence_b_max() runs a ghost refill + a syncing device reduction, so
        # sample it only on cadence steps, the final step (for divb_linf_final),
        # and when a progress log needs it -- not every step.
        on_cadence = cadence > 0 and step_done % cadence == 0
        need_divb = deck.diagnostics.divb and (
            on_cadence or is_last
            or (log_every > 0 and step_done % log_every == 0))
        divb_now = float(solver.divergence_b_max()) if need_divb else None
        if divb_now is not None:
            divb_series.append(divb_now)
        if on_cadence:
            snapshot = {
                "step": step_done,
                "time_s": sim_time,
                "fields": _snapshot_fields(solver, deck),
            }
            if deck.diagnostics.divb:
                snapshot["divb"] = divb_now
            snapshots.append(snapshot)
        if log_every > 0 and step_done % log_every == 0:
            elapsed = time.time() - t0
            rate = step_done / elapsed if elapsed > 0 else 0.0
            remaining = (deck.time.steps - step_done) / rate if rate > 0 else float("nan")
            message = (f"step {step_done}/{deck.time.steps}  "
                       f"t={sim_time:.6e}  rate={rate:.0f} step/s  "
                       f"eta={remaining:.0f}s")
            if deck.diagnostics.divb:
                message += f"  |divB|inf={divb_now:.3e}"
            print(message, flush=True)

    flat = _flatten_for_npz(
        solver, deck, step_done, sim_time, divb_series, snapshots,
        extra_scalars)
    flat.update(initial_state)
    # Passing a suffixless path to np.savez silently writes to ``path + .npz``.
    # Open the already-confined path so diagnostics.output_path is exact.
    with out_path.open("wb") as stream:
        np.savez(stream, **flat)
    return step_done, sim_time


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
    run.add_argument("--devices", default=None, metavar="auto|ID[,ID...]",
                     help="Eligible node-local GPU pool (activates distributed execution).")
    run.add_argument("--decomposition", default=None, metavar="auto|PXxPY",
                     help="Virtual GPU tile decomposition.")
    run.add_argument("--transport", choices=("auto", "staged", "direct"),
                     default=None, help="Inter-process halo transport policy.")
    run.add_argument("--diagnostics-layout", choices=("gathered", "sharded"),
                     default=None, help="Distributed diagnostics file layout.")
    run.add_argument("--checkpoint", default=None, metavar="PATH",
                     help="Write the final collective HDF5 checkpoint to PATH.")
    run.add_argument("--checkpoint-every", type=_positive_int, default=None,
                     metavar="N", help="Replace --checkpoint at absolute step multiples of N.")
    run.add_argument("--restart", default=None, metavar="PATH",
                     help="Restart from a collective Quasar HDF5 checkpoint.")
    run.set_defaults(func=_do_run)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (_DistributedUsageError,
            _distributed.DistributedUnavailableError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
