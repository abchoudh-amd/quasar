#!/usr/bin/env python3
"""Generate the PIC deck for an H+/mu- run in a square-toroid magnet.

The toroid's symmetry axis is **z_lab** (matches examples/square_toroid).
The PIC z=0 plane is therefore the EQUATORIAL slice through the donut
bore — PIC (x, y) ↔ lab (x_L, y_L) at z_L=0. The dominant external
field at this slice is **Bz** (axial in the lab; out-of-plane for PIC),
which drives in-plane cyclotron gyration of the species.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

# Load the package's single CFL helper directly from its file (without importing
# the quasar package, which would require the compiled _core extension) so this
# standalone generator and the CLI share one formula.
_numerics_path = (Path(__file__).resolve().parents[2]
                  / "python" / "quasar" / "pic" / "numerics.py")
_spec = importlib.util.spec_from_file_location("_quasar_pic_numerics", _numerics_path)
_numerics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_numerics)
_cfl_dt = _numerics.cfl_dt


R0_M = 0.10
SIDE_M = 0.04
N_SHEET_FILAMENTS = 64
SHEET_CURRENT_A = 1000.0
N_SEGMENTS = 128

N_TF_COILS = 16
TF_COIL_CURRENT_A = 1000.0
TF_R_INNER_M = R0_M - SIDE_M / 2.0
TF_R_OUTER_M = R0_M + SIDE_M / 2.0
TF_Z_MIN_M = -SIDE_M / 2.0
TF_Z_MAX_M = +SIDE_M / 2.0


PIC_NX = 128
PIC_NY = 128
PIC_LX_M = SIDE_M
PIC_LY_M = SIDE_M
PIC_ORIGIN_X_M = R0_M - PIC_LX_M / 2.0
PIC_ORIGIN_Y_M = -PIC_LY_M / 2.0

TOTAL_TIME_S = 1.0e-6
_DX = PIC_LX_M / PIC_NX
_DY = PIC_LY_M / PIC_NY
DT_CFL_S = _cfl_dt(_DX, _DY)  # fdtd_order=2 default; matches the CLI's auto dt
STEPS = int(math.ceil(TOTAL_TIME_S / DT_CFL_S))
CADENCE = 0  # In-memory field snapshots disabled; use --write-every for rolling out.npz checkpoint.

N_PARTICLES = 20000
DENSITY_PER_M3 = 1.0e15
TEMPERATURE_EV = 10.0

BORE_CENTER_X_M = R0_M
BORE_CENTER_Y_M = 0.0
BLOCK_HALF_WIDTH_M = PIC_LX_M / 4.0
BLOCK_X_MIN_M = BORE_CENTER_X_M - BLOCK_HALF_WIDTH_M
BLOCK_X_MAX_M = BORE_CENTER_X_M + BLOCK_HALF_WIDTH_M
BLOCK_Y_MIN_M = BORE_CENTER_Y_M - BLOCK_HALF_WIDTH_M
BLOCK_Y_MAX_M = BORE_CENTER_Y_M + BLOCK_HALF_WIDTH_M

Q_E = 1.602176634e-19
M_PROTON = 1.67262192369e-27
M_MUON = 1.883531627e-28


def _fmt(x: float) -> str:
    return f"{x:+.8f}"


def _fmt_time(t_s: float) -> str:
    if t_s < 1e-6:
        return f"{t_s*1e9:.3f} ns"
    if t_s < 1e-3:
        return f"{t_s*1e6:.3f} us"
    if t_s < 1.0:
        return f"{t_s*1e3:.3f} ms"
    return f"{t_s:.3f} s"


def _tf_coil_block(*, name: str, theta_rad: float,
                   current_A: float) -> list[str]:
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
        f"      - name: {name}",
        f"        current_A: {_fmt(current_A)}",
        "        geometry:",
        "          type: polyline",
        "          points_xyz_m:",
    ]
    for (x, y, z) in pts:
        lines.append(f"            - [{_fmt(x)}, {_fmt(y)}, {_fmt(z)}]")
    lines.append("")
    return lines


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
        "# the donut bore. PIC sees the bore as a ring in (x_pic, y_pic).",
        "#",
        "# Magnet = axisymmetric square-cross-section current sheets",
        f"#   + {N_TF_COILS} discrete rectangular TF coils ({TF_COIL_CURRENT_A:.0f} A each).",
        "# Sheets give poloidal B (B_x, B_z); TF coils give toroidal",
        "# B_phi ~ mu_0 N I / (2 pi R) ~ 0.032 T at R = R0.",
        "#",
        f"# Particles are loaded into a {2*BLOCK_HALF_WIDTH_M:.3f} m square block",
        f"# centered on the right-side bore at (x, y) = ({BORE_CENTER_X_M:.3f},",
        f"# {BORE_CENTER_Y_M:.3f}) m -- the toroid's half-width x half-height",
        f"# footprint. Total physical time = {_fmt_time(TOTAL_TIME_S)}",
        f"# ({STEPS} CFL steps at dt ~ {DT_CFL_S:.3e} s).",
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

    lines.append("      # Discrete TF coils (rectangular, bore-hugging)")
    for k in range(N_TF_COILS):
        theta = 2.0 * math.pi * k / N_TF_COILS
        lines.extend(_tf_coil_block(name=f"tf_coil_{k:02d}",
                                     theta_rad=theta,
                                     current_A=+TF_COIL_CURRENT_A))

    lines.extend([
        "species:",
        "  - name: H+",
        f"    charge_C: {+Q_E:.12e}",
        f"    mass_kg: {M_PROTON:.12e}",
        f"    n_particles: {N_PARTICLES}",
        "    initial:",
        "      distribution: maxwellian_block",
        f"      density_per_m3: {DENSITY_PER_M3:.6e}",
        f"      temperature_eV: {TEMPERATURE_EV:.3f}",
        "      region:",
        f"        x_min_m: {BLOCK_X_MIN_M:.6f}",
        f"        x_max_m: {BLOCK_X_MAX_M:.6f}",
        f"        y_min_m: {BLOCK_Y_MIN_M:.6f}",
        f"        y_max_m: {BLOCK_Y_MAX_M:.6f}",
        "  - name: mu-",
        f"    charge_C: {-Q_E:.12e}",
        f"    mass_kg: {M_MUON:.12e}",
        f"    n_particles: {N_PARTICLES}",
        "    initial:",
        "      distribution: maxwellian_block",
        f"      density_per_m3: {DENSITY_PER_M3:.6e}",
        f"      temperature_eV: {TEMPERATURE_EV:.3f}",
        "      region:",
        f"        x_min_m: {BLOCK_X_MIN_M:.6f}",
        f"        x_max_m: {BLOCK_X_MAX_M:.6f}",
        f"        y_min_m: {BLOCK_Y_MIN_M:.6f}",
        f"        y_max_m: {BLOCK_Y_MAX_M:.6f}",
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
        "boundary:",
        "  # Mark particles that drift out of the PIC domain as dead, so we",
        "  # can count loss. Default (omitted) is 'periodic' = no-op.",
        "  particle: absorbing",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    out = Path(__file__).with_name("input.yaml")
    out.write_text(build_yaml(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
