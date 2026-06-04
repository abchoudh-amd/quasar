#!/usr/bin/env python3
"""Generate the PIC deck for an H+/mu- run in a 1 m square-toroid magnet,
simulated on the POLOIDAL cross-section (PIC plane = lab x-z).

Unlike the sibling examples/square_toroid_pic (which simulates the equatorial
x-y slice), this deck sets ``plane: xz`` so the 2D grid is the lab y=0 meridional
cut. The torus axis is lab z, so that cut shows the 0.3 m square cross-section:
PIC x = major radius R, PIC y = lab z (the cross-section's vertical). The
out-of-plane direction is lab y, along which the toroidal field B_phi (from the
TF coils) points -- so B_phi is the confining out-of-plane field the PIC sees as
external Bz.

Run ``python build_yaml.py`` to (re)write input.yaml. Run with ``--field-check``
to also emit field_check.yaml: a quasar.coil eval deck with the SAME conductors
sampled on the lab y=0 plane, for computing the magnet field before the PIC run.
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

# Shared deck-emitting helpers (float formatter + circular-loop block), loaded by
# file path for the same reason as the numerics helper above.
_deckgen_path = Path(__file__).resolve().parents[1] / "_deckgen.py"
_deckgen_spec = importlib.util.spec_from_file_location(
    "_quasar_example_deckgen", _deckgen_path)
_deckgen = importlib.util.module_from_spec(_deckgen_spec)
_deckgen_spec.loader.exec_module(_deckgen)
_fmt = _deckgen.fmt_float


R0_M = 1.0
SIDE_M = 0.30
N_SHEET_FILAMENTS = 64
# Current per sheet group (top / bottom / inner cylinder / outer cylinder),
# spread over N_SHEET_FILAMENTS filaments each.
SHEET_CURRENT_A = 150_000.0
N_SEGMENTS = 128

N_TF_COILS = 8
TF_COIL_CURRENT_A = 25_000.0
TF_R_INNER_M = R0_M - SIDE_M / 2.0
TF_R_OUTER_M = R0_M + SIDE_M / 2.0
TF_Z_MIN_M = -SIDE_M / 2.0
TF_Z_MAX_M = +SIDE_M / 2.0


PIC_NX = 128
PIC_NY = 128
# Simulate 90% of the cross-section width in both dimensions, centered on the bore.
PIC_FRACTION = 0.90
PIC_LX_M = PIC_FRACTION * SIDE_M
PIC_LY_M = PIC_FRACTION * SIDE_M
# PIC x = lab x (major radius R); PIC y = lab z (cross-section vertical). The
# window is centered on the right-side bore at (x, z) = (R0, 0).
PIC_ORIGIN_X_M = R0_M - PIC_LX_M / 2.0
PIC_ORIGIN_Y_M = -PIC_LY_M / 2.0

TOTAL_TIME_S = 5.0e-6
OUTPUT_PATH = "out.npz"


def _recompute_derived() -> None:
    """Recompute resolution-dependent globals (cell size, CFL dt, step count) from
    the current PIC_NX/PIC_NY. STEPS is resolution-dependent (dt shrinks as the
    grid refines), so this must run after any --nx override so the emitted deck's
    `steps` still encodes TOTAL_TIME_S."""
    global _DX, _DY, DT_CFL_S, STEPS
    _DX = PIC_LX_M / PIC_NX
    _DY = PIC_LY_M / PIC_NY
    DT_CFL_S = _cfl_dt(_DX, _DY)  # fdtd_order=2 default; matches the CLI's auto dt
    STEPS = int(math.ceil(TOTAL_TIME_S / DT_CFL_S))


_recompute_derived()
CADENCE = 0  # In-memory field snapshots disabled; use --write-every for rolling out.npz checkpoint.

N_PARTICLES = 20000
DENSITY_PER_M3 = 1.0e15
TEMPERATURE_EV = 1.0

# Particles fill half the cross-section: a centered block whose side is half the
# cross-section width (so its area is 1/4 the full a x a, matching the sibling's
# "centered 50% block" convention measured per-dimension).
BORE_CENTER_X_M = R0_M
BORE_CENTER_Y_M = 0.0
BLOCK_HALF_WIDTH_M = SIDE_M / 4.0
BLOCK_X_MIN_M = BORE_CENTER_X_M - BLOCK_HALF_WIDTH_M
BLOCK_X_MAX_M = BORE_CENTER_X_M + BLOCK_HALF_WIDTH_M
BLOCK_Y_MIN_M = BORE_CENTER_Y_M - BLOCK_HALF_WIDTH_M
BLOCK_Y_MAX_M = BORE_CENTER_Y_M + BLOCK_HALF_WIDTH_M

Q_E = 1.602176634e-19
M_PROTON = 1.67262192369e-27
M_MUON = 1.883531627e-28


def _fmt_time(t_s: float) -> str:
    if t_s < 1e-6:
        return f"{t_s*1e9:.3f} ns"
    if t_s < 1e-3:
        return f"{t_s*1e6:.3f} us"
    if t_s < 1.0:
        return f"{t_s*1e3:.3f} ms"
    return f"{t_s:.3f} s"


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
    leading ``- name:`` sits at column ``indent``. Shared by the PIC deck
    (indent=6, nested under external_field.evaluator) and the coil field-check
    deck (indent=2, top-level conductors:) so both decks use identical magnets."""
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


def build_yaml() -> str:
    lines = [
        "# H+ / mu- plasma in the POLOIDAL cross-section of a 1 m square-toroid.",
        "#",
        "# Torus axis = lab z. plane: xz makes the PIC grid the lab y=0 meridional",
        "# cut, i.e. the 0.30 m square cross-section: PIC x = major radius R,",
        "# PIC y = lab z. The out-of-plane direction is lab y, along which the",
        "# toroidal field B_phi (from the TF coils) points -- the PIC sees it as",
        "# external Bz and it provides the cross-section confinement.",
        "#",
        "# Magnet = axisymmetric square-cross-section current sheets",
        f"#   ({SHEET_CURRENT_A:.0f} A per sheet group)",
        f"#   + {N_TF_COILS} discrete rectangular TF coils ({TF_COIL_CURRENT_A:.0f} A each).",
        f"# B_phi ~ mu_0 N I / (2 pi R) ~ "
        f"{4e-7*math.pi*N_TF_COILS*TF_COIL_CURRENT_A/(2*math.pi*R0_M):.3f} T at R = R0.",
        "#",
        f"# PIC domain = {PIC_FRACTION*100:.0f}% of the {SIDE_M:.2f} m cross-section "
        f"({PIC_LX_M:.3f} m sq),",
        f"# centered on the bore at (x, z) = ({BORE_CENTER_X_M:.3f}, "
        f"{BORE_CENTER_Y_M:.3f}) m.",
        f"# Particles fill a centered {2*BLOCK_HALF_WIDTH_M:.3f} m square block",
        f"# (half the cross-section width). Temperature {TEMPERATURE_EV:.3f} eV.",
        f"# Total physical time = {_fmt_time(TOTAL_TIME_S)} "
        f"({STEPS} CFL steps at dt ~ {DT_CFL_S:.3e} s).",
        "#",
        "# Regenerate with:",
        "#   python examples/square_toroid_pic_1m/build_yaml.py",
        "",
        "units: SI",
        "",
        "# Simulate the poloidal cross-section (lab x-z), not the equatorial slice.",
        "plane: xz",
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
    ]
    lines.extend(_conductor_lines(indent=6))

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
        f"  output_path: {OUTPUT_PATH}",
        f"  cadence: {CADENCE}",
        "  fields: [bz, ex, ey]",
        "  per_species: true",
        "",
        "boundary:",
        "  # Mark particles that drift out of the PIC domain as dead, so we",
        "  # can count loss against the magnet wall. Default (omitted) is periodic.",
        "  particle: absorbing",
        "",
    ])
    return "\n".join(lines)


def build_field_check_yaml() -> str:
    """A quasar.coil eval deck: the SAME magnet conductors sampled on the lab
    y=0 plane (the plane the PIC samples in xz mode), over the PIC cross-section
    window. Used to compute/verify the magnet field before the PIC run."""
    lines = [
        "# Field-check deck: magnet field on the lab y=0 plane (the PIC xz slice).",
        "# Same conductors as input.yaml. B_xyz_grid has shape (nz, ny, nx, 3) with",
        "# ny=1; out-of-plane B_phi is the |B.y| component.",
        "#",
        "# Run with:",
        "#   python -m quasar.coil.cli run examples/square_toroid_pic_1m/field_check.yaml",
        "",
        "units: SI",
        "",
        "conductors:",
    ]
    lines.extend(_conductor_lines(indent=2))
    lines.extend([
        "observation:",
        "  type: grid",
        "  # x spans the radial window; y fixed at 0 (the PIC slice plane);",
        "  # z spans the cross-section vertical. Matches the PIC domain extent.",
        "  bounds_m:",
        f"    - [{PIC_ORIGIN_X_M:.6f}, {PIC_ORIGIN_X_M + PIC_LX_M:.6f}]",
        "    - [0.0, 0.0]",
        f"    - [{PIC_ORIGIN_Y_M:.6f}, {PIC_ORIGIN_Y_M + PIC_LY_M:.6f}]",
        f"  resolution: [{PIC_NX}, 1, {PIC_NY}]",
        "",
        "output:",
        "  format: npz",
        "  path: field_check.npz",
        "  fields:",
        "    - B_xyz",
        "    - B_xyz_grid",
        "    - B_magnitude",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    import argparse
    global PIC_NX, PIC_NY, OUTPUT_PATH
    p = argparse.ArgumentParser(description="Generate the square_toroid_pic_1m deck.")
    p.add_argument("--nx", type=int, default=None,
                   help="Grid resolution (sets both nx and ny); default 128. "
                        "STEPS is recomputed so the deck still targets 5 us.")
    p.add_argument("--out", default="input.yaml",
                   help="Output deck filename (written next to this script).")
    p.add_argument("--output-path", default=None,
                   help="diagnostics.output_path written into the deck "
                        "(default: out.npz).")
    p.add_argument("--field-check", action="store_true",
                   help="Also emit field_check.yaml (coil eval deck).")
    args = p.parse_args()

    if args.nx is not None:
        PIC_NX = args.nx
        PIC_NY = args.nx
        _recompute_derived()
    if args.output_path is not None:
        OUTPUT_PATH = args.output_path

    out = Path(__file__).with_name(args.out)
    out.write_text(build_yaml(), encoding="utf-8")
    print(f"wrote {out}  (nx=ny={PIC_NX}, steps={STEPS}, output_path={OUTPUT_PATH})")
    if args.field_check:
        fc = Path(__file__).with_name("field_check.yaml")
        fc.write_text(build_field_check_yaml(), encoding="utf-8")
        print(f"wrote {fc}")


if __name__ == "__main__":
    main()
