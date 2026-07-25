"""Plot the final state of a square_toroid_pic run from out.npz.

Reads the rolling-checkpoint out.npz (no resim) and renders:
  - particle_loss.png : alive-count vs time for each species
  - final_state.png   : final positions over external Bz background
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from quasar._plotting import require_pyplot
from quasar.pic.postprocess import (
    field_periodicity,
    species_names,
    yee_component_coordinates,
    yee_component_view,
)

plt = require_pyplot(agg=True)

HERE = Path(__file__).resolve().parent
NPZ = HERE / "out.npz"
DECK = HERE / "input.yaml"


def _domain_extent(deck: dict) -> tuple[float, float, float, float]:
    d = deck["domain"]
    x_lo = d.get("origin_x_m", 0.0)
    y_lo = d.get("origin_y_m", 0.0)
    return x_lo, x_lo + d["lx_m"], y_lo, y_lo + d["ly_m"]


def plot_loss(data, deck: dict) -> Path:
    t = data["series_time_s"] * 1e6  # µs
    fig, ax = plt.subplots(figsize=(8, 5))
    for sp in species_names(data):
        key = f"series_alive_{sp}"
        if key not in data.files:
            continue
        alive = data[key]
        n0 = alive[0] if len(alive) else 0
        ax.plot(t, alive, label=f"{sp}  (start={n0}, end={int(alive[-1])})")
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("alive particle count")
    ax.set_title(f"Particle loss vs time  "
                 f"(t_final = {float(data['final_time_s'].item()):.3e} s)")
    ax.grid(alpha=0.3)
    ax.legend()
    out = HERE / "particle_loss.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_final_state(data, deck: dict) -> Path:
    x_lo, x_hi, y_lo, y_hi = _domain_extent(deck)
    nx = int(data["nx"].item())
    ny = int(data["ny"].item())
    nghost = int(data["nghost"].item()) if "nghost" in data.files else 0
    geometry = (str(data["geometry"].item())
                if "geometry" in data.files else "cartesian")
    periodic_x, periodic_y = field_periodicity(data)
    bz = yee_component_view(
        data["external_bz"], nx, ny, nghost, "bz", geometry,
        periodic_x=periodic_x, periodic_y=periodic_y)
    bz_x, bz_y = yee_component_coordinates(
        "bz", geometry, nx, ny, x_lo, y_lo, x_hi - x_lo, y_hi - y_lo,
        periodic_x=periodic_x, periodic_y=periodic_y,
        view_shape=bz.shape)
    species = species_names(data)

    fig, axes = plt.subplots(1, len(species), figsize=(6 * len(species), 6),
                              sharex=True, sharey=True, squeeze=False)
    bz_max = float(np.max(np.abs(bz))) or 1.0
    for ax, sp in zip(axes[0], species):
        im = ax.pcolormesh(
            bz_x, bz_y, bz, shading="nearest", cmap="RdBu_r",
            vmin=-bz_max, vmax=bz_max, alpha=0.7)
        x = data[f"species_{sp}_x"]
        y = data[f"species_{sp}_y"]
        alive = data[f"species_{sp}_alive"].astype(bool)
        ax.scatter(x[alive], y[alive], s=0.3, c="k", alpha=0.5,
                   label=f"alive ({int(alive.sum())})")
        dead = ~alive
        if dead.any():
            ax.scatter(x[dead], y[dead], s=0.3, c="orange", alpha=0.5,
                       label=f"lost ({int(dead.sum())})")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{sp} at t = {float(data['final_time_s'].item()):.3e} s")
        ax.legend(markerscale=10, loc="upper right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="external B_z (T)")
    fig.suptitle("square_toroid_pic: final positions on external B_z")
    fig.tight_layout()
    out = HERE / "final_state.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> int:
    with open(DECK, encoding="utf-8") as fh:
        deck = yaml.safe_load(fh)
    with np.load(NPZ, allow_pickle=False) as data:
        print(f"loaded {NPZ}  "
              f"(final_time = {float(data['final_time_s'].item()):.3e} s)")
        print(f"wrote {plot_loss(data, deck)}")
        print(f"wrote {plot_final_state(data, deck)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
