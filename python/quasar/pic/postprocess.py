"""Plot helpers for PIC ``out.npz`` snapshots written by ``quasar.pic.cli``.

Two entry points:

* ``plot(npz_path, out_dir)``   — write field heatmaps + per-species particle
  scatter PNGs alongside the input.
* ``main(argv)``                — ``python -m quasar.pic.postprocess
  examples/square_toroid_pic/out.npz``.

matplotlib is imported lazily so the rest of the package stays usable on
machines without it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


def rms(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def reshape_with_ghost(flat: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Return the interior ``(ny, nx)`` view of a ghost-padded Yee field buffer.

    The buffer storage is ``(nx + 2g) * (ny + 2g)`` for some ghost width ``g``
    (1 for 2nd-order FDTD, 2 for 4th-order). The ghost width is recovered from the
    flat size rather than assumed, then ``g`` cells are stripped from each side. A
    buffer already sized ``nx * ny`` (no ghosts) is reshaped directly."""
    g = 0
    while (nx + 2 * (g + 1)) * (ny + 2 * (g + 1)) <= flat.size:
        g += 1
    if flat.size == (nx + 2 * g) * (ny + 2 * g) and g > 0:
        return flat.reshape(ny + 2 * g, nx + 2 * g)[g:-g, g:-g]
    return flat.reshape(ny, nx)


# Backwards-compatible alias (older callers used the private name).
_reshape_with_ghost = reshape_with_ghost


def plot(npz_path: Path | str, out_dir: Path | str | None = None) -> list[Path]:
    """Render diagnostic PNGs for an ``out.npz``. Returns written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npz_path = Path(npz_path)
    out_dir = Path(out_dir) if out_dir is not None else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=False)
    nx = int(data["nx"][0])
    ny = int(data["ny"][0])
    written: list[Path] = []

    ext_bz = _reshape_with_ghost(data["external_bz"], nx, ny)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(ext_bz, origin="lower", aspect="equal", cmap="RdBu_r")
    ax.set_title("external Bz (T)")
    ax.set_xlabel("x cell")
    ax.set_ylabel("y cell")
    fig.colorbar(im, ax=ax)
    p = out_dir / f"{npz_path.stem}_external_bz.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)

    for fname in ("field_bz", "field_ex", "field_ey"):
        if fname not in data.files:
            continue
        arr = _reshape_with_ghost(data[fname], nx, ny)
        vmax = float(np.max(np.abs(arr))) or 1.0
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(arr, origin="lower", aspect="equal", cmap="RdBu_r",
                       vmin=-vmax, vmax=+vmax)
        ax.set_title(fname)
        fig.colorbar(im, ax=ax)
        p = out_dir / f"{npz_path.stem}_{fname}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    species_names = sorted({
        k[len("species_"):-len("_x")]
        for k in data.files
        if k.startswith("species_") and k.endswith("_x")
    })
    if species_names:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(np.abs(ext_bz), origin="lower", aspect="equal",
                  cmap="Greys", alpha=0.4)
        colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
        for i, sp in enumerate(species_names):
            x = data[f"species_{sp}_x"]
            y = data[f"species_{sp}_y"]
            alive = data[f"species_{sp}_alive"].astype(bool)
            ax.scatter(x[alive], y[alive], s=0.5,
                       c=colors[i % len(colors)], label=sp, alpha=0.4)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("particle positions")
        ax.legend(markerscale=10)
        p = out_dir / f"{npz_path.stem}_particles.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    return written


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="quasar.pic.postprocess",
                                description="Render PIC out.npz diagnostics.")
    p.add_argument("npz", help="Path to out.npz produced by quasar.pic.cli.")
    p.add_argument("--out-dir", default=None,
                   help="Directory to write PNGs into (default: alongside npz).")
    args = p.parse_args(argv)
    written = plot(args.npz, args.out_dir)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
