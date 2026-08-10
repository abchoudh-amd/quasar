#!/usr/bin/env python3
"""Generate the ideal-MHD deck for a square-toroid bore on the POLOIDAL slice,
with the confining poloidal B supplied as a Biot-Savart coil FIELD-SPLIT
background B0.

Coordinate mapping (same poloidal cut as examples/square_toroid_pic_1m):
  MHD x = lab X = major radius R,  MHD y = lab Z (cross-section vertical),
  out-of-plane = lab Y, along which the toroidal guide field Bz points.

Why field-split: the inline conductors are evaluated on the solver-derived full
padded corner grid and supplied as a static background B0 (B = B0 + b), so the
field is never replaced by a generic boundary extrapolation. The evolving
perturbation b starts at zero and is advanced by constrained transport. The
90%-bore box is an artificial crop through the vacuum field, not a conducting
surface: the coil field has nonzero normal component there, so both fluid and
perturbation-field boundaries are open. The toroidal guide field is a uniform
out-of-plane bz in the state.

The one emitted ``input.yaml`` deck contains both the SI, axisymmetric
cylindrical (R,Z) MHD run and the square-toroid conductor geometry:

  PYTHONPATH=build/hip-gfx942-release/python \\
    python -m quasar.mhd.cli run examples/square_toroid_mhd/input.yaml

Regenerate the deck with: python examples/square_toroid_mhd/build_yaml.py
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

# --- MHD grid: the bore interior (90% of the cross-section), centered on R0 ----
MHD_NX = 128
MHD_NY = 128
MHD_FRACTION = 0.90
MHD_LX_M = MHD_FRACTION * SIDE_M
MHD_LY_M = MHD_FRACTION * SIDE_M
MHD_ORIGIN_X_M = R0_M - MHD_LX_M / 2.0
MHD_ORIGIN_Y_M = -MHD_LY_M / 2.0
# Toroidal (out-of-plane) guide field. The deck uses SI; b_scale is a
# dimensionless optional amplitude multiplier for the generated A in T*m.
BZ_TOROIDAL = 0.1
B_SCALE = 1.0

# Confined plasma blob (denser/higher-p inside a centered square).  The sampled
# coil reaches roughly 0.6 T, or O(1e5 Pa) magnetic pressure.  Pressures of
# 1/0.1 Pa make the discontinuous blob beta O(1e-6), below the truncation scale
# of the supported cylindrical high-order update; the conservative positivity
# controller then correctly refuses to cross zero internal energy.  These
# values retain a strongly magnetized beta O(1e-3--1e-2) case while remaining
# resolved for the shipped grid and full 400-step run.
RHO_IN = 10.0
RHO_OUT = 1.0
P_IN = 1000.0
P_OUT = 100.0
BLOB_HALF_M = 0.25 * MHD_LX_M

GAMMA = 1.6666667
STEPS = 400
OUTPUT_PATH = "out.npz"


def _conductor_lines(indent: int) -> list[str]:
    """Emit the axisymmetric current-sheet filaments indented so the
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
    # Poloidal TF windings generate the toroidal guide field represented by
    # initial.bz below. They are deliberately absent here: interpreting their
    # discrete 3-D A_y slice as axisymmetric A_phi would create a spurious
    # poloidal background and double-count the guide system.
    return lines


def build_mhd_yaml() -> str:
    bphi = 4e-7 * math.pi * N_TF_COILS * TF_COIL_CURRENT_A / (2 * math.pi * R0_M)
    lines = [
        "# Ideal-MHD: confined plasma blob in a square-toroid bore (poloidal slice).",
        "#",
        "# MHD x = lab X = major radius R; MHD y = lab Z (cross-section vertical);",
        "# out-of-plane (bz) = lab Y, the toroidal guide-field direction.",
        "#",
        "# The confining POLOIDAL field is a field-split background B0 (B = B0 + b)",
        "# built directly from the inline conductors. The loader evaluates lab-Y",
        "# vector potential A on the solver-derived padded cell-corner grid and",
        "# differences it with the annular curl to a face-staggered field",
        "#   b0x_face(i,j) = -(A[j+1,i]-A[j,i])/dy   (B_R, left face)",
        "#   b0y_face(i,j) = (R_hi*A[j,i+1]-R_lo*A[j,i])/int(R dR)",
        "#                                                   (B_z, bottom face)",
        "# which is divergence-free by construction and static (never ghost-refilled).",
        "# The crop cuts through this field, so its edges are open rather than",
        "# conducting walls; outflow is applied consistently to the fluid and the",
        "# evolving perturbation b. CT preserves div(b). bz is a uniform",
        f"# toroidal guide field ({BZ_TOROIDAL}); B_phi ~ mu0 N I/(2 pi R) ~ "
        f"{bphi:.3f} T from the TF coils.",
        "#",
        "# Regenerate: python examples/square_toroid_mhd/build_yaml.py",
        "",
        "units: SI",
        f"domain: {{nx: {MHD_NX}, ny: {MHD_NY}, lx_m: {MHD_LX_M:.6f}, "
        f"ly_m: {MHD_LY_M:.6f}, origin_x_m: {MHD_ORIGIN_X_M:.6f}, "
        f"origin_y_m: {MHD_ORIGIN_Y_M:.6f}}}",
        "geometry: cylindrical",
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
        "  conductors:",
    ]
    lines.extend(_conductor_lines(indent=4))
    lines.extend([
        "  bz0: 0.0               # toroidal field lives in the state (initial.bz)",
        "  params:",
        f"    b_scale: {B_SCALE}       # dimensionless multiplier of SI A (T*m)",
        "    vacuum_project: true  # fixed-boundary discrete annular vacuum solve",
        f"time: {{dt_s: auto, steps: {STEPS}}}",
        "diagnostics: {output_path: out.npz, cadence: 0,",
        "              fields: [rho, mx, my, mz, energy, bx, by, bz], divb: true}",
        "boundary: {fluid: [outflow, outflow, outflow, outflow],",
        "           field: [outflow, outflow, outflow, outflow]}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    import argparse
    global MHD_NX, MHD_NY
    p = argparse.ArgumentParser(
        description="Generate the square_toroid_mhd deck.")
    p.add_argument("--nx", type=int, default=None,
                   help="Grid resolution (sets both nx and ny); default 128.")
    args = p.parse_args()
    if args.nx is not None:
        MHD_NX = args.nx
        MHD_NY = args.nx

    here = Path(__file__).parent
    (here / "input.yaml").write_text(build_mhd_yaml(), encoding="utf-8")
    print(f"wrote {here/'input.yaml'} (nx=ny={MHD_NX}, steps={STEPS})")


if __name__ == "__main__":
    main()
