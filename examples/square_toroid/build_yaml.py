#!/usr/bin/env python3
"""Generate the square-cross-section toroidal current-sheet example deck."""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Shared deck-emitting helpers, loaded by file path so this standalone generator
# does not import the quasar package (which would require the compiled _core).
_deckgen_path = Path(__file__).resolve().parents[1] / "_deckgen.py"
_spec = importlib.util.spec_from_file_location("_quasar_example_deckgen", _deckgen_path)
_deckgen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_deckgen)
_fmt = _deckgen.fmt_float


R0_M = 0.10
SIDE_M = 0.04
N_SHEET_FILAMENTS = 256
SHEET_CURRENT_A = 1500.0
N_SEGMENTS = 256

OBS_X_MIN_M = 0.085
OBS_X_MAX_M = 0.115
OBS_Z_MIN_M = -0.015
OBS_Z_MAX_M = 0.015
OBS_NX = 257
OBS_NZ = 257


def _loop_block(
    *,
    name: str,
    current_A: float,
    center_z_m: float,
    radius_m: float,
) -> list[str]:
    return _deckgen.loop_block(
        indent=2, name=name, current_A=current_A,
        center_z_m=center_z_m, radius_m=radius_m, n_segments=N_SEGMENTS)


def build_yaml() -> str:
    half_side = SIDE_M / 2.0
    inner_radius = R0_M - half_side
    outer_radius = R0_M + half_side
    delta = SIDE_M / N_SHEET_FILAMENTS
    filament_current = SHEET_CURRENT_A / N_SHEET_FILAMENTS

    lines = [
        "# Square-cross-section toroidal magnet modeled as four toroidal current sheets.",
        "#",
        "# Geometry:",
        f"#   R0 = {R0_M:.6f} m, square side a = {SIDE_M:.6f} m",
        f"#   inner radius = {inner_radius:.6f} m, outer radius = {outer_radius:.6f} m",
        f"#   top/bottom z = +/-{half_side:.6f} m",
        "#",
        "# Currents:",
        "#   top and bottom faces carry +phi current",
        "#   inner and outer cylindrical faces carry -phi current",
        f"#   each sheet has total current {SHEET_CURRENT_A:.6f} A",
        f"#   each sheet is represented by {N_SHEET_FILAMENTS} circular-loop filaments",
        "#",
        "# Regenerate with:",
        "#   python examples/square_toroid/build_yaml.py",
        "",
        "units: SI",
        "",
        "conductors:",
        "  # Top sheet (+phi), z = +a/2",
    ]

    radii = [
        inner_radius + (k + 0.5) * delta
        for k in range(N_SHEET_FILAMENTS)
    ]
    heights = [
        -half_side + (k + 0.5) * delta
        for k in range(N_SHEET_FILAMENTS)
    ]

    for k, radius_m in enumerate(radii):
        lines.extend(
            _loop_block(
                name=f"top_sheet_{k:02d}",
                current_A=+filament_current,
                center_z_m=+half_side,
                radius_m=radius_m,
            )
        )

    lines.append("  # Bottom sheet (+phi), z = -a/2")
    for k, radius_m in enumerate(radii):
        lines.extend(
            _loop_block(
                name=f"bottom_sheet_{k:02d}",
                current_A=+filament_current,
                center_z_m=-half_side,
                radius_m=radius_m,
            )
        )

    lines.append("  # Inner cylinder (-phi), R = R0 - a/2")
    for k, z_m in enumerate(heights):
        lines.extend(
            _loop_block(
                name=f"inner_cylinder_{k:02d}",
                current_A=-filament_current,
                center_z_m=z_m,
                radius_m=inner_radius,
            )
        )

    lines.append("  # Outer cylinder (-phi), R = R0 + a/2")
    for k, z_m in enumerate(heights):
        lines.extend(
            _loop_block(
                name=f"outer_cylinder_{k:02d}",
                current_A=-filament_current,
                center_z_m=z_m,
                radius_m=outer_radius,
            )
        )

    lines.extend(
        [
            "observation:",
            "  type: plane",
            f"  origin_xyz:   [{_fmt(OBS_X_MIN_M)}, 0.0, {_fmt(OBS_Z_MIN_M)}]",
            "  u_axis_xyz:   [1.0, 0.0, 0.0]",
            "  v_axis_xyz:   [0.0, 0.0, 1.0]",
            f"  u_extent_m:   {OBS_X_MAX_M - OBS_X_MIN_M:.8f}",
            f"  v_extent_m:   {OBS_Z_MAX_M - OBS_Z_MIN_M:.8f}",
            f"  nu: {OBS_NX}",
            f"  nv: {OBS_NZ}",
            "",
            "output:",
            "  format: npz",
            "  path: out.npz",
            "  fields:",
            "    - B_xyz",
            "    - B_magnitude",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    out = Path(__file__).with_name("input.yaml")
    out.write_text(build_yaml(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
