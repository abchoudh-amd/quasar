#!/usr/bin/env python3
"""Generate BOTH decks for the square-frame magnetic quadrupole.

A 30 cm square frame of z-directed current filaments (current flows along the
lab z-axis, i.e. PERPENDICULAR to the 30 cm plane):

    top & bottom rows  -> +z current   (y = +/-0.15 m, wires spread along x)
    left & right rows  -> -z current   (x = +/-0.15 m, wires spread along y)

With this +/+/-/- sign pattern the four rows form a magnetic QUADRUPOLE: in the
z=0 midplane the field is purely transverse (B_x, B_y), zero on the central axis,
rising ~linearly outward. The 0.25 cm PIC patch sits on that central null.

The magnet geometry is defined exactly once (``_wire_specs``) and emitted into
two sibling decks:

  * examples/square_quad_field/input.yaml  -- coil "field calculator": the 256
    wires + an observation grid over the PIC patch (quasar.coil.cli).
  * examples/square_quad_pic/input.yaml    -- EM-PIC run: the same 256 wires as
    the external Biot-Savart field, with H+ and mu- species (quasar.pic.cli).

Regenerate with:
    python examples/square_quad_field/build_yaml.py
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

# Load the package's single CFL helper directly from its file (without importing
# the quasar package, which would require the compiled _core extension) so this
# standalone generator and the CLI share one formula. Mirrors the loader pattern
# in examples/square_toroid_pic/build_yaml.py.
_numerics_path = (Path(__file__).resolve().parents[2]
                  / "python" / "quasar" / "pic" / "numerics.py")
_spec = importlib.util.spec_from_file_location("_quasar_pic_numerics", _numerics_path)
_numerics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_numerics)
_cfl_dt = _numerics.cfl_dt

# Shared float formatter, loaded by file path for the same reason.
_deckgen_path = Path(__file__).resolve().parents[1] / "_deckgen.py"
_deckgen_spec = importlib.util.spec_from_file_location(
    "_quasar_example_deckgen", _deckgen_path)
_deckgen = importlib.util.module_from_spec(_deckgen_spec)
_deckgen_spec.loader.exec_module(_deckgen)
_fmt = _deckgen.fmt_float


# --- Magnet geometry ---------------------------------------------------------
HALF_M = 0.15                 # square frame half-size (30 cm side)
N_PER_SIDE = 64               # filaments per side (256 total)
TOTAL_CURRENT_A = 15000.0     # per-side total current (15 kA)
PER_WIRE_A = TOTAL_CURRENT_A / N_PER_SIDE  # 234.375 A
ZHALF_M = 1.0                 # wires span z in [-1, +1] m (~ infinite at the patch)

# --- PIC patch ---------------------------------------------------------------
PIC_NX = 128
PIC_NY = 128
PIC_L_M = 0.0025              # 0.25 cm square domain
PIC_ORIGIN_M = -PIC_L_M / 2.0  # centered on the magnet null at the origin

_DX = PIC_L_M / PIC_NX
_DY = PIC_L_M / PIC_NY
DT_CFL_S = _cfl_dt(_DX, _DY)   # fdtd_order=2 default; matches the CLI's auto dt
STEPS = 10000                  # ~0.23 ns physical (see README for the 1 us scaling)
CADENCE = 500                  # 20 in-memory field snapshots

PPC = 100
N_PARTICLES = PIC_NX * PIC_NY * PPC  # 1,638,400 per species
DENSITY_PER_M3 = 1.0e18
TEMPERATURE_EV = 10000.0       # 10 keV (Maxwellian temperature)

# Coil field-map observation grid = the PIC patch (z=0 midplane).
OBS_LO_M = PIC_ORIGIN_M
OBS_HI_M = PIC_ORIGIN_M + PIC_L_M

Q_E = 1.602176634e-19
M_PROTON = 1.67262192369e-27
M_MUON = 1.883531627e-28


def _wire_specs() -> list[tuple[str, float, float, float]]:
    """The 256 z-directed wires as (name, current_A, x_m, y_m).

    Wires are cell-centered along each 30 cm side (offset by half a spacing) so
    the four rows never double-occupy a corner with opposite-sign currents.
    Current sign encodes direction: +z (top/bottom), -z (left/right). Each wire's
    polyline runs -z -> +z, so +current_A => +z current.
    """
    delta = (2.0 * HALF_M) / N_PER_SIDE
    coords = [-HALF_M + (k + 0.5) * delta for k in range(N_PER_SIDE)]
    specs: list[tuple[str, float, float, float]] = []
    for k, x in enumerate(coords):
        specs.append((f"top_{k:02d}", +PER_WIRE_A, x, +HALF_M))
    for k, x in enumerate(coords):
        specs.append((f"bottom_{k:02d}", +PER_WIRE_A, x, -HALF_M))
    for k, y in enumerate(coords):
        specs.append((f"left_{k:02d}", -PER_WIRE_A, -HALF_M, y))
    for k, y in enumerate(coords):
        specs.append((f"right_{k:02d}", -PER_WIRE_A, +HALF_M, y))
    return specs


def _polyline_block(*, indent: int, name: str, current_A: float,
                    x_m: float, y_m: float) -> list[str]:
    """A z-directed straight-wire ``polyline`` conductor block.

    ``indent`` is the column of the leading ``- name:`` item (2 for a top-level
    ``conductors:`` list, 6 nested under ``external_field.evaluator``)."""
    pad = " " * indent
    inner = " " * (indent + 2)
    return [
        f"{pad}- name: {name}",
        f"{pad}  current_A: {_fmt(current_A)}",
        f"{pad}  geometry:",
        f"{inner}  type: polyline",
        f"{inner}  points_xyz_m:",
        f"{inner}    - [{_fmt(x_m)}, {_fmt(y_m)}, {_fmt(-ZHALF_M)}]",
        f"{inner}    - [{_fmt(x_m)}, {_fmt(y_m)}, {_fmt(+ZHALF_M)}]",
        "",
    ]


def _conductor_lines(indent: int) -> list[str]:
    lines: list[str] = []
    for name, current_A, x_m, y_m in _wire_specs():
        lines.extend(_polyline_block(indent=indent, name=name,
                                     current_A=current_A, x_m=x_m, y_m=y_m))
    return lines


def build_field_yaml() -> str:
    """Coil deck: the 256 wires + an observation grid over the PIC patch."""
    lines = [
        "# Magnetic-field calculator for a 30 cm square-frame quadrupole.",
        "#",
        "# 256 z-directed filaments (current along +/- lab z): top & bottom",
        f"#   rows = +z, left & right rows = -z, {PER_WIRE_A:.3f} A each",
        f"#   ({TOTAL_CURRENT_A:.0f} A per side). In the z=0 midplane the field is",
        "#   transverse (B_x, B_y), zero at the center, rising outward.",
        "#",
        "# Observation grid = the 0.25 cm PIC patch at z=0.",
        "#",
        "# Regenerate with:",
        "#   python examples/square_quad_field/build_yaml.py",
        "",
        "units: SI",
        "",
        "conductors:",
    ]
    lines.extend(_conductor_lines(indent=2))
    lines.extend([
        "observation:",
        "  type: grid",
        f"  bounds_m: [[{_fmt(OBS_LO_M)}, {_fmt(OBS_HI_M)}], "
        f"[{_fmt(OBS_LO_M)}, {_fmt(OBS_HI_M)}], [0.0, 0.0]]",
        f"  resolution: [{PIC_NX}, {PIC_NY}, 1]",
        "",
        "output:",
        "  format: npz",
        "  path: out.npz",
        "  fields: [B_xyz, B_magnitude]",
        "",
    ])
    return "\n".join(lines)


def build_pic_yaml() -> str:
    """PIC deck: the same 256 wires as the external field; H+/mu- at 10 keV."""
    lines = [
        "# EM-PIC: H+ / mu- plasma on the central null of a square quadrupole.",
        "#",
        "# The 30 cm square frame of z-directed filaments (top/bottom +z,",
        "# left/right -z) imposes a transverse (B_x, B_y) quadrupole field on a",
        f"# {PIC_L_M*100:.2f} cm patch centered on the magnetic null. Because B is",
        "# in-plane and the slice is 2D-xy, q v x B couples in-plane velocity to",
        "# the out-of-plane v_z (transverse focusing, not in-plane gyration).",
        "#",
        f"# {PIC_NX}x{PIC_NY} cells, {PPC} ppc, two species -> "
        f"{2*N_PARTICLES:,} particles.",
        f"# steps={STEPS} ~ {STEPS*DT_CFL_S*1e9:.3f} ns at dt ~ {DT_CFL_S:.3e} s.",
        "# A full 1 us study needs ~{:d} CFL steps (explicit-EM light-crossing"
        .format(int(math.ceil(1.0e-6 / DT_CFL_S))),
        "#   limit, not particle dynamics) -- see README before scaling up.",
        "#",
        "# Regenerate with:",
        "#   python examples/square_quad_field/build_yaml.py",
        "",
        "units: SI",
        "",
        "domain:",
        f"  nx: {PIC_NX}",
        f"  ny: {PIC_NY}",
        f"  lx_m: {PIC_L_M:.6f}",
        f"  ly_m: {PIC_L_M:.6f}",
        f"  origin_x_m: {PIC_ORIGIN_M:.6f}",
        f"  origin_y_m: {PIC_ORIGIN_M:.6f}",
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
        "  fields: [ex, ey, ez, bx, by, bz]",
        "  per_species: true",
        "",
        "boundary:",
        "  # Absorbing particle walls (count loss off the open patch). Field walls",
        "  # are PEC because first-order Mur 'outflow' is not charge/current compatible",
        "  # and the solver rejects charged species with a Mur field boundary.",
        "  # Self-fields are tiny relative to the static external quadrupole here.",
        "  particle: [absorbing, absorbing, absorbing, absorbing]",
        "  field: [pec, pec, pec, pec]",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    field_dir = Path(__file__).resolve().parent
    pic_dir = field_dir.parent / "square_quad_pic"
    pic_dir.mkdir(parents=True, exist_ok=True)

    field_out = field_dir / "input.yaml"
    pic_out = pic_dir / "input.yaml"
    field_out.write_text(build_field_yaml(), encoding="utf-8")
    pic_out.write_text(build_pic_yaml(), encoding="utf-8")
    print(f"wrote {field_out}")
    print(f"wrote {pic_out}")


if __name__ == "__main__":
    main()
