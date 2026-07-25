"""Plot initial vs final particle positions for the square_toroid_pic deck.

Loads the deck, seeds species (recording initial positions), runs the PIC
solver, and renders a 2x2 figure: initial vs final positions for H+ and mu-.
Titles report start / alive / lost counts per species.

This view needs the *initial* positions, which the rolling ``out.npz`` does not
store, so it builds and steps a solver via the public ``quasar.pic.prepare_run``
seam (it is not a duplicate of ``plot_solution.py``, which reads ``out.npz``).
The figure is written to ``initial_vs_final.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from quasar._plotting import require_pyplot
from quasar.pic import Units, prepare_run
from quasar.pic import io as pic_io

HERE = Path(__file__).resolve().parent
DECK = HERE / "input.yaml"


def main() -> int:
    deck = pic_io.load(DECK)
    units = Units(deck)
    solver, indices, dt, dt_si = prepare_run(deck, units, seed=0)

    initial = []
    for idx in indices:
        host = solver.species_at(idx).to_host()
        initial.append({
            "name": deck.species[idx].name,
            "x": np.array(host["x"], copy=True),
            "y": np.array(host["y"], copy=True),
            "alive": np.array(host["alive"], copy=True).astype(bool),
        })

    step_cap = deck.time.steps
    t_end = deck.time.t_end_s
    print(f"running at most {step_cap} steps at dt={dt_si:.3e} s ...", flush=True)
    report_every = max(1, step_cap // 10)
    sim_time = 0.0
    steps_done = 0
    while steps_done < step_cap and (t_end is None or sim_time < t_end):
        dt_step_si = dt_si
        clipped = False
        if t_end is not None:
            remaining = t_end - sim_time
            dt_step_si = min(dt_step_si, remaining)
            clipped = dt_step_si == remaining
        if not np.isfinite(dt_step_si) or dt_step_si <= 0.0:
            raise RuntimeError("PIC plotting run timestep made no forward progress")
        dt_step = dt if dt_step_si == dt_si else units.time(dt_step_si)
        solver.step(dt_step)
        sim_time = t_end if clipped else sim_time + dt_step_si
        steps_done += 1
        if steps_done % report_every == 0 or clipped:
            print(f"  step {steps_done}/{step_cap}", flush=True)
    solver.finalize()

    finals = []
    for idx in indices:
        host = solver.species_at(idx).to_host()
        finals.append({
            "name": deck.species[idx].name,
            "x": np.array(host["x"], copy=True),
            "y": np.array(host["y"], copy=True),
            "alive": np.array(host["alive"], copy=True).astype(bool),
        })

    plt = require_pyplot(agg=True)

    domain = deck.domain
    x_lo, x_hi = domain.origin_x_m, domain.origin_x_m + domain.lx_m
    y_lo, y_hi = domain.origin_y_m, domain.origin_y_m + domain.ly_m

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    for row, sp in enumerate(zip(initial, finals)):
        init, fin = sp
        n_start = init["alive"].sum()
        n_alive_end = fin["alive"].sum()
        n_lost = int(n_start - n_alive_end)

        ax = axes[row, 0]
        m = init["alive"]
        ax.scatter(init["x"][m], init["y"][m], s=0.5, c="tab:blue", alpha=0.4)
        ax.set_title(f"{init['name']} initial: {int(n_start)} particles")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        ax = axes[row, 1]
        m_alive = fin["alive"]
        m_dead = ~fin["alive"]
        ax.scatter(fin["x"][m_alive], fin["y"][m_alive], s=0.5,
                   c="tab:blue", alpha=0.4, label=f"alive: {int(n_alive_end)}")
        if m_dead.any():
            ax.scatter(fin["x"][m_dead], fin["y"][m_dead], s=0.5,
                       c="tab:red", alpha=0.4, label=f"lost: {n_lost}")
        ax.set_title(f"{fin['name']} after {steps_done} steps: "
                     f"alive={int(n_alive_end)}, lost={n_lost}")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.legend(markerscale=10, loc="upper right")

        print(f"{init['name']:>4s}: start={int(n_start)} "
              f"alive_end={int(n_alive_end)} lost={n_lost}")

    fig.suptitle(f"square_toroid_pic: {steps_done} steps, "
                 f"t_final = {sim_time:.3e} s")
    fig.tight_layout()
    out = HERE / "initial_vs_final.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
