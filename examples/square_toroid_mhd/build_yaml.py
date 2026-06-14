#!/usr/bin/env python3
"""Generate the ideal-MHD deck for a square-toroid bore on the POLOIDAL slice,
with the confining poloidal B supplied as a Biot-Savart coil FIELD-SPLIT
background B0.

Coordinate mapping (same poloidal cut as examples/square_toroid_pic_1m):
  MHD x = lab X = major radius R,  MHD y = lab Z (cross-section vertical),
  out-of-plane = lab Y, along which the toroidal guide field Bz points.

Why field-split: the constrained-transport solver freezes the discrete div(B) at
its seeded value, and no open boundary condition can reproduce the exterior coil
field -- so seeding the coil field into the evolving STATE corrupts div(B) at the
boundary ring and floods inward. Instead the static, curl-free coil field is a
field-split background B0 (B = B0 + b): B0 is never ghost-refilled, and the
evolving perturbation b starts at zero (div(b) = 0 exactly, forever). The
toroidal guide field is a uniform out-of-plane bz in the state.

Two synced decks are emitted:

  * ``coil.yaml`` — the square-toroid current sheets + TF coils sampled on the
    lab Y=0 plane at the cell-corner grid of the FULL PADDED MHD domain (interior
    + nghost ghost layers), writing ``A_xyz_grid`` (the vector potential). The
    background loader differences the corner lab-Y A into face-staggered B0 that
    is discretely divergence-free by construction, defined into the ghosts.
  * ``input.yaml`` — the MHD run: a confined plasma blob in the bore on a uniform
    toroidal guide field bz, with the coil poloidal field as background_field.a_file.

Workflow (the MHD background reads coil.npz, so run the coil deck first):

  PYTHONPATH=build/hip-gfx942-release/python \\
    python -m quasar.coil.cli run examples/square_toroid_mhd/coil.yaml
  PYTHONPATH=build/hip-gfx942-release/python \\
    python -m quasar.mhd.cli run examples/square_toroid_mhd/input.yaml

Regenerate the decks with: python examples/square_toroid_mhd/build_yaml.py

The coil grid spans the PADDED domain, so it depends on the reconstruction halo
(NGHOST): mp7 needs nghost=4. Keep NGHOST in sync with numerics.reconstruction.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

# Shared deck-emitting helpers (float formatter + circular-loop block), loaded by
# file path so this standalone generator needs no compiled _core extension.
_deckgen_path = Path(__file__).resolve().parents[1] / "_deckgen.py"
_deckgen_spec = importlib.util.spec_from_file_location(
    "_quasar_example_deckgen", _deckgen_path)
_deckgen = importlib.util.module_from_spec(_deckgen_spec)
_deckgen_spec.loader.exec_module(_deckgen)
_fmt = _deckgen.fmt_float

# --- magnet geometry (square-cross-section toroid, R0 = 1 m, 0.30 m bore) -----
R0_M = 1.0
SIDE_M = 0.30
N_SHEET_FILAMENTS = 64
SHEET_CURRENT_A = 150_000.0
N_SEGMENTS = 128

N_TF_COILS = 8
TF_COIL_CURRENT_A = 25_000.0
TF_R_INNER_M = R0_M - SIDE_M / 2.0
TF_R_OUTER_M = R0_M + SIDE_M / 2.0
TF_Z_MIN_M = -SIDE_M / 2.0
TF_Z_MAX_M = +SIDE_M / 2.0

# --- MHD grid: the bore interior (90% of the cross-section), centered on R0 ----
MHD_NX = 128
MHD_NY = 128
MHD_FRACTION = 0.90
MHD_LX_M = MHD_FRACTION * SIDE_M
MHD_LY_M = MHD_FRACTION * SIDE_M
MHD_ORIGIN_X_M = R0_M - MHD_LX_M / 2.0
MHD_ORIGIN_Y_M = -MHD_LY_M / 2.0
# Reconstruction halo: mp7 -> nghost=4 (mp5 -> 3, muscl -> 2). The coil A grid
# spans the padded domain so B0 is defined into the ghost layers; keep in sync
# with numerics.reconstruction below.
NGHOST = 4

# Toroidal (out-of-plane) guide field and the poloidal-field normalization. The
# coil A is in SI (T*m); b_scale maps it into the deck's normalized field units.
BZ_TOROIDAL = 0.1
B_SCALE = 1.0

# Confined plasma blob (denser/higher-p inside a centered square).
RHO_IN = 10.0
RHO_OUT = 1.0
P_IN = 1.0
P_OUT = 0.1
BLOB_HALF_M = 0.25 * MHD_LX_M

GAMMA = 1.6666667
STEPS = 400
OUTPUT_PATH = "out.npz"


def _tf_coil_block(*, indent: int, name: str, theta_rad: float,
                   current_A: float) -> list[str]:
    pad = " " * indent
    inner = " " * (indent + 4)
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    corners_Rz = [
        (TF_R_INNER_M, TF_Z_MIN_M),
        (TF_R_OUTER_M, TF_Z_MIN_M),
        (TF_R_OUTER_M, TF_Z_MAX_M),
        (TF_R_INNER_M, TF_Z_MAX_M),
        (TF_R_INNER_M, TF_Z_MIN_M),
    ]
    pts = [(R * cos_t, R * sin_t, z) for (R, z) in corners_Rz]
    lines = [
        f"{pad}- name: {name}",
        f"{pad}  current_A: {_fmt(current_A)}",
        f"{pad}  geometry:",
        f"{inner}type: polyline",
        f"{inner}points_xyz_m:",
    ]
    for (x, y, z) in pts:
        lines.append(f"{inner}  - [{_fmt(x)}, {_fmt(y)}, {_fmt(z)}]")
    lines.append("")
    return lines


def _conductor_lines(indent: int) -> list[str]:
    """Emit the full conductor list (current sheets + TF coils) indented so the
    leading ``- name:`` sits at column ``indent``."""
    half_side = SIDE_M / 2.0
    inner_radius = R0_M - half_side
    outer_radius = R0_M + half_side
    delta = SIDE_M / N_SHEET_FILAMENTS
    filament_current = SHEET_CURRENT_A / N_SHEET_FILAMENTS

    radii = [inner_radius + (k + 0.5) * delta for k in range(N_SHEET_FILAMENTS)]
    heights = [-half_side + (k + 0.5) * delta for k in range(N_SHEET_FILAMENTS)]

    def loop(name, current_A, center_z_m, radius_m):
        return _deckgen.loop_block(
            indent=indent, name=name, current_A=current_A,
            center_z_m=center_z_m, radius_m=radius_m, n_segments=N_SEGMENTS)

    pad = " " * indent
    lines: list[str] = [f"{pad}# Top sheet (+phi), z = +a/2"]
    for k, radius_m in enumerate(radii):
        lines.extend(loop(f"top_sheet_{k:02d}", +filament_current,
                          +half_side, radius_m))
    lines.append(f"{pad}# Bottom sheet (+phi), z = -a/2")
    for k, radius_m in enumerate(radii):
        lines.extend(loop(f"bottom_sheet_{k:02d}", +filament_current,
                          -half_side, radius_m))
    lines.append(f"{pad}# Inner cylinder (-phi), R = R0 - a/2")
    for k, y_m in enumerate(heights):
        lines.extend(loop(f"inner_cylinder_{k:02d}", -filament_current,
                          y_m, inner_radius))
    lines.append(f"{pad}# Outer cylinder (-phi), R = R0 + a/2")
    for k, y_m in enumerate(heights):
        lines.extend(loop(f"outer_cylinder_{k:02d}", -filament_current,
                          y_m, outer_radius))
    lines.append(f"{pad}# Discrete TF coils (rectangular, bore-hugging)")
    for k in range(N_TF_COILS):
        theta = 2.0 * math.pi * k / N_TF_COILS
        lines.extend(_tf_coil_block(indent=indent, name=f"tf_coil_{k:02d}",
                                    theta_rad=theta, current_A=+TF_COIL_CURRENT_A))
    return lines


def build_coil_yaml() -> str:
    """Coil eval deck: vector potential A on the lab Y=0 cell-corner grid of the
    FULL PADDED MHD domain (interior + NGHOST ghost layers each side).

    The padded domain has Nx = nx + 2*NGHOST cells over
    [origin_x - NGHOST*dx, origin_x + lx + NGHOST*dx] (and likewise in z); its
    corner grid is (Nx+1) x (Ny+1). Sampling A there lets the background loader
    define a divergence-free B0 into the ghost layers the reconstruction stencil
    reads. A_xyz_grid has shape (Ny+1, 1, Nx+1, 3); the loader differences its
    lab-Y component (index 1) into a face-staggered, discretely div-free B0."""
    dx = MHD_LX_M / MHD_NX
    dy = MHD_LY_M / MHD_NY
    npx = MHD_NX + 2 * NGHOST
    npy = MHD_NY + 2 * NGHOST
    x_lo = MHD_ORIGIN_X_M - NGHOST * dx
    x_hi = MHD_ORIGIN_X_M + MHD_LX_M + NGHOST * dx
    z_lo = MHD_ORIGIN_Y_M - NGHOST * dy
    z_hi = MHD_ORIGIN_Y_M + MHD_LY_M + NGHOST * dy
    lines = [
        "# Square-toroid magnet: vector potential A on the lab Y=0 poloidal slice,",
        "# sampled at the cell-corner grid of the PADDED MHD domain (interior +",
        f"# NGHOST={NGHOST} ghost layers). resolution = [Nx+1, 1, Ny+1] with",
        f"# Nx = nx+2*NGHOST = {npx}, Ny = ny+2*NGHOST = {npy}. A_xyz_grid has shape",
        "# (Ny+1, 1, Nx+1, 3); the MHD background_field.a_file loader differences its",
        "# lab-Y component into a divergence-free face-staggered poloidal B0.",
        "#",
        "# Run BEFORE the MHD deck (which reads the produced coil.npz):",
        "#   python -m quasar.coil.cli run examples/square_toroid_mhd/coil.yaml",
        "",
        "units: SI",
        "",
        "conductors:",
    ]
    lines.extend(_conductor_lines(indent=2))
    lines.extend([
        "observation:",
        "  type: grid",
        "  # x = radial window (corners), y fixed at 0 (the poloidal slice plane),",
        "  # z = cross-section vertical (corners). Padded-domain corner grid.",
        "  bounds_m:",
        f"    - [{x_lo:.6f}, {x_hi:.6f}]",
        "    - [0.0, 0.0]",
        f"    - [{z_lo:.6f}, {z_hi:.6f}]",
        f"  resolution: [{npx + 1}, 1, {npy + 1}]",
        "",
        "output:",
        "  format: npz",
        "  path: coil.npz",
        "  fields:",
        "    - A_xyz_grid",
        "",
    ])
    return "\n".join(lines)


def build_mhd_yaml() -> str:
    bphi = 4e-7 * math.pi * N_TF_COILS * TF_COIL_CURRENT_A / (2 * math.pi * R0_M)
    lines = [
        "# Ideal-MHD: confined plasma blob in a square-toroid bore (poloidal slice).",
        "#",
        "# MHD x = lab X = major radius R; MHD y = lab Z (cross-section vertical);",
        "# out-of-plane (bz) = lab Y, the toroidal guide-field direction.",
        "#",
        "# The confining POLOIDAL field is a field-split background B0 (B = B0 + b)",
        "# built from coil.npz: the lab-Y vector potential A on the padded cell-corner",
        "# grid, differenced to a face-staggered",
        "#   b0x_face(i,j) = -(A[j+1,i]-A[j,i])/dy   (B_R, left face)",
        "#   b0y_face(i,j) =  (A[j,i+1]-A[j,i])/dx   (B_z, bottom face)",
        "# which is divergence-free by construction and static (never ghost-refilled,",
        "# so no open BC can corrupt it). The evolving perturbation b starts at zero,",
        "# so the constrained-transport div(b) stays exactly zero. bz is a uniform",
        f"# toroidal guide field ({BZ_TOROIDAL}); B_phi ~ mu0 N I/(2 pi R) ~ "
        f"{bphi:.3f} T from the TF coils.",
        "#",
        "# Run coil.yaml FIRST to produce coil.npz, then this deck.",
        "# Regenerate: python examples/square_toroid_mhd/build_yaml.py",
        "",
        "units: normalized",
        f"domain: {{nx: {MHD_NX}, ny: {MHD_NY}, lx_m: {MHD_LX_M:.6f}, "
        f"ly_m: {MHD_LY_M:.6f}, origin_x_m: {MHD_ORIGIN_X_M:.6f}, "
        f"origin_y_m: {MHD_ORIGIN_Y_M:.6f}}}",
        "geometry: cartesian",
        "numerics: {gamma: 1.6666667, reconstruction: mp7, riemann: hlld, "
        "integrator: ssprk3, ct: fd_ct_christlieb, positivity: troubled_cell, "
        "rho_floor: 1e-8, p_floor: 1e-9, cfl: 0.2}",
        "initial:",
        "  type: confined_blob",
        "  params:",
        f"    bz: {BZ_TOROIDAL}             # uniform toroidal guide field (in the state)",
        f"    rho_in: {RHO_IN}",
        f"    rho_out: {RHO_OUT}",
        f"    p_in: {P_IN}",
        f"    p_out: {P_OUT}",
        f"    blob_half: {BLOB_HALF_M:.6f}   # half-width of the centered plasma blob",
        "background_field:",
        "  enabled: true",
        "  a_file: coil.npz       # padded corner-grid A_xyz_grid from coil.yaml",
        "  bz0: 0.0               # toroidal field lives in the state (initial.bz)",
        f"  params: {{b_scale: {B_SCALE}}}   # SI A (T*m) -> normalized field units",
        f"time: {{dt_s: auto, steps: {STEPS}}}",
        "diagnostics: {output_path: out.npz, cadence: 0,",
        "              fields: [rho, mx, my, mz, energy, bx, by, bz], divb: true}",
        "boundary: {fluid: [wall, wall, wall, wall],",
        "           field: [wall, wall, wall, wall]}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    import argparse
    global MHD_NX, MHD_NY
    p = argparse.ArgumentParser(
        description="Generate the square_toroid_mhd decks.")
    p.add_argument("--nx", type=int, default=None,
                   help="Grid resolution (sets both nx and ny); default 128.")
    args = p.parse_args()
    if args.nx is not None:
        MHD_NX = args.nx
        MHD_NY = args.nx

    here = Path(__file__).parent
    (here / "coil.yaml").write_text(build_coil_yaml(), encoding="utf-8")
    (here / "input.yaml").write_text(build_mhd_yaml(), encoding="utf-8")
    print(f"wrote {here/'coil.yaml'} and {here/'input.yaml'} "
          f"(nx=ny={MHD_NX}, steps={STEPS})")


if __name__ == "__main__":
    main()
