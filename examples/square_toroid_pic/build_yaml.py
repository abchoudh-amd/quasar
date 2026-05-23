#!/usr/bin/env python3
"""Generate the PIC deck for an H+/mu- run in a square-toroid magnet.

The toroid's symmetry axis is **z_lab** (matches examples/square_toroid).
The PIC z=0 plane is therefore the EQUATORIAL slice through the donut
bore — PIC (x, y) ↔ lab (x_L, y_L) at z_L=0. The dominant external
field at this slice is **Bz** (axial in the lab; out-of-plane for PIC),
which drives in-plane cyclotron gyration of the species.
"""

from __future__ import annotations

from pathlib import Path


R0_M = 0.10
SIDE_M = 0.04
N_SHEET_FILAMENTS = 64
SHEET_CURRENT_A = 5000.0
N_SEGMENTS = 128


PIC_NX = 128
PIC_NY = 128
PIC_LX_M = 0.30
PIC_LY_M = 0.30
PIC_ORIGIN_X_M = -PIC_LX_M / 2.0
PIC_ORIGIN_Y_M = -PIC_LY_M / 2.0

STEPS = 2000
CADENCE = 200

N_PARTICLES = 20000
DENSITY_PER_M3 = 1.0e15
TEMPERATURE_EV = 10.0

Q_E = 1.602176634e-19
M_PROTON = 1.67262192369e-27
M_MUON = 1.883531627e-28


def _fmt(x: float) -> str:
    return f"{x:+.8f}"


def _loop_block(*, name: str, current_A: float, center_z_m: float,
                radius_m: float) -> list[str]:
    return [
        f"      - name: {name}",
        f"        current_A: {_fmt(current_A)}",
        "        geometry:",
        "          type: circular_loop",
        f"          center_xyz: [0.0, 0.0, {_fmt(center_z_m)}]",
        "          axis_xyz:   [0.0, 0.0, 1.0]",
        f"          radius_m: {_fmt(radius_m)}",
        f"          n_segments: {N_SEGMENTS}",
        "",
    ]


def build_yaml() -> str:
    half_side = SIDE_M / 2.0
    inner_radius = R0_M - half_side
    outer_radius = R0_M + half_side
    delta = SIDE_M / N_SHEET_FILAMENTS
    filament_current = SHEET_CURRENT_A / N_SHEET_FILAMENTS

    radii = [inner_radius + (k + 0.5) * delta for k in range(N_SHEET_FILAMENTS)]
    heights = [-half_side + (k + 0.5) * delta for k in range(N_SHEET_FILAMENTS)]

    lines = [
        "# H+ / mu- plasma in the equatorial slice of a square-toroid magnet.",
        "#",
        "# Toroid axis = z_lab; PIC z=0 plane is the equatorial slice through",
        "# the donut bore. PIC sees the bore as a ring in (x_pic, y_pic), with",
        "# external Bz (out-of-plane) driving in-plane cyclotron gyration.",
        "#",
        "# Regenerate with:",
        "#   python examples/square_toroid_pic/build_yaml.py",
        "",
        "units: SI",
        "",
        "domain:",
        f"  nx: {PIC_NX}",
        f"  ny: {PIC_NY}",
        f"  lx_m: {PIC_LX_M:.6f}",
        f"  ly_m: {PIC_LY_M:.6f}",
        f"  origin_x_m: {PIC_ORIGIN_X_M:.6f}",
        f"  origin_y_m: {PIC_ORIGIN_Y_M:.6f}",
        "",
        "numerics:",
        "  fdtd_order: 2",
        "  shape: cic",
        "",
        "external_field:",
        "  evaluator:",
        "    type: biot_savart",
        "    conductors:",
        "      # Top sheet (+phi), z = +a/2",
    ]

    for k, radius_m in enumerate(radii):
        lines.extend(_loop_block(name=f"top_sheet_{k:02d}",
                                  current_A=+filament_current,
                                  center_z_m=+half_side, radius_m=radius_m))

    lines.append("      # Bottom sheet (+phi), z = -a/2")
    for k, radius_m in enumerate(radii):
        lines.extend(_loop_block(name=f"bottom_sheet_{k:02d}",
                                  current_A=+filament_current,
                                  center_z_m=-half_side, radius_m=radius_m))

    lines.append("      # Inner cylinder (-phi), R = R0 - a/2")
    for k, y_m in enumerate(heights):
        lines.extend(_loop_block(name=f"inner_cylinder_{k:02d}",
                                  current_A=-filament_current,
                                  center_z_m=y_m, radius_m=inner_radius))

    lines.append("      # Outer cylinder (-phi), R = R0 + a/2")
    for k, y_m in enumerate(heights):
        lines.extend(_loop_block(name=f"outer_cylinder_{k:02d}",
                                  current_A=-filament_current,
                                  center_z_m=y_m, radius_m=outer_radius))

    lines.extend([
        "species:",
        "  - name: H+",
        f"    charge_C: {+Q_E:.12e}",
        f"    mass_kg: {M_PROTON:.12e}",
        f"    n_particles: {N_PARTICLES}",
        "    initial:",
        "      distribution: maxwellian_uniform",
        f"      density_per_m3: {DENSITY_PER_M3:.6e}",
        f"      temperature_eV: {TEMPERATURE_EV:.3f}",
        "  - name: mu-",
        f"    charge_C: {-Q_E:.12e}",
        f"    mass_kg: {M_MUON:.12e}",
        f"    n_particles: {N_PARTICLES}",
        "    initial:",
        "      distribution: maxwellian_uniform",
        f"      density_per_m3: {DENSITY_PER_M3:.6e}",
        f"      temperature_eV: {TEMPERATURE_EV:.3f}",
        "",
        "time:",
        "  dt_s: auto",
        f"  steps: {STEPS}",
        "",
        "diagnostics:",
        "  output_path: out.npz",
        f"  cadence: {CADENCE}",
        "  fields: [bz, ex, ey]",
        "  per_species: true",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    out = Path(__file__).with_name("input.yaml")
    out.write_text(build_yaml(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
