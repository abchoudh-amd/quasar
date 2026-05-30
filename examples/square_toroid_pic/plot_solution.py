"""Plot the final state of a square_toroid_pic run from out.npz.

Reads the rolling-checkpoint out.npz (no resim) and renders:
  - particle_loss.png : alive-count vs time for each species
  - final_state.png   : final positions over external Bz background
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
NPZ = HERE / "out.npz"
DECK = HERE / "input.yaml"


def _domain_extent(deck: dict) -> tuple[float, float, float, float]:
    d = deck["domain"]
    x_lo = d.get("origin_x_m", 0.0)
    y_lo = d.get("origin_y_m", 0.0)
    return x_lo, x_lo + d["lx_m"], y_lo, y_lo + d["ly_m"]


def _species_names(data) -> list[str]:
    return sorted({k.split("_", 1)[1].rsplit("_", 1)[0]
                   for k in data.files if k.startswith("species_")})


def plot_loss(data, deck: dict) -> Path:
    t = data["series_time_s"] * 1e6  # µs
    fig, ax = plt.subplots(figsize=(8, 5))
    for sp in _species_names(data):
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
    nx, ny = int(data["nx"].item()), int(data["ny"].item())
    bz_full = data["external_bz"]
    # Fields are stored on a Yee grid with a 1-cell ghost halo on each side.
    side = int(round(bz_full.size ** 0.5))
    bz = bz_full.reshape(side, side)
    if side == ny + 2:
        bz = bz[1:-1, 1:-1]
    species = _species_names(data)

    fig, axes = plt.subplots(1, len(species), figsize=(6 * len(species), 6),
                              sharex=True, sharey=True, squeeze=False)
    bz_max = float(np.max(np.abs(bz))) or 1.0
    for ax, sp in zip(axes[0], species):
        im = ax.imshow(bz, origin="lower", extent=(x_lo, x_hi, y_lo, y_hi),
                       cmap="RdBu_r", vmin=-bz_max, vmax=bz_max, alpha=0.7)
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
    data = np.load(NPZ)
    with open(DECK) as fh:
        deck = yaml.safe_load(fh)
    print(f"loaded {NPZ}  (final_time = {float(data['final_time_s'].item()):.3e} s)")
    print(f"wrote {plot_loss(data, deck)}")
    print(f"wrote {plot_final_state(data, deck)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
