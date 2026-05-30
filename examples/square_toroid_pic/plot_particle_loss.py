"""Plot initial vs final particle positions for the square_toroid_pic deck.

Loads the deck, seeds species (recording initial positions), runs the PIC
solver, and renders a 2x2 figure: initial vs final positions for H+ and mu-.
Titles report start / alive / lost counts per species.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DECK = HERE / "input.yaml"


def main() -> int:
    from quasar.pic import cli as pic_cli
    from quasar.pic import io as pic_io

    deck = pic_io.load(DECK)
    solver = pic_cli._make_solver(deck)
    pic_cli._apply_external_field(solver, deck)
    rng = np.random.default_rng(0)
    indices = pic_cli._seed_species(solver, deck, rng)
    dt = (pic_cli._cfl_dt(deck.domain)
          if deck.time.dt_s == "auto" else float(deck.time.dt_s))

    initial = []
    for idx in indices:
        host = solver.species_at(idx).to_host()
        initial.append({
            "name": deck.species[idx].name,
            "x": np.array(host["x"], copy=True),
            "y": np.array(host["y"], copy=True),
            "alive": np.array(host["alive"], copy=True).astype(bool),
        })

    steps = deck.time.steps
    print(f"running {steps} steps at dt={dt:.3e} s ...", flush=True)
    report_every = max(1, steps // 10)
    for s in range(steps):
        solver.step(dt)
        if (s + 1) % report_every == 0:
            print(f"  step {s + 1}/{steps}", flush=True)

    finals = []
    for idx in indices:
        host = solver.species_at(idx).to_host()
        finals.append({
            "name": deck.species[idx].name,
            "x": np.array(host["x"], copy=True),
            "y": np.array(host["y"], copy=True),
            "alive": np.array(host["alive"], copy=True).astype(bool),
        })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
        ax.set_title(f"{fin['name']} after {steps} steps: "
                     f"alive={int(n_alive_end)}, lost={n_lost}")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.legend(markerscale=10, loc="upper right")

        print(f"{init['name']:>4s}: start={int(n_start)} "
              f"alive_end={int(n_alive_end)} lost={n_lost}")

    fig.suptitle(f"square_toroid_pic: {steps} steps, "
                 f"t_final = {steps * dt:.3e} s")
    fig.tight_layout()
    out = HERE / "particle_loss.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
