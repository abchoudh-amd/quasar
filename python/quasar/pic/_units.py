"""Unit conversion between an SI PIC deck and the solver's internal units.

The C++ FDTD / Boris-push / Esirkepov-deposit kernels operate in natural units
(``c = eps0 = mu0 = 1``). A ``units: SI`` deck must therefore be non-dimensionalized
before it reaches the solver, otherwise an SI timestep drives field kernels that
assume ``c = 1`` and the electromagnetic evolution is physically wrong.

This module wraps the C++ :class:`quasar.pic.Normalization` (length ``c/omega_p``,
time ``1/omega_p``, velocity ``c``, with charge/mass/field/density scales) and
exposes linear conversion factors that work on scalars and NumPy arrays alike.
For ``units: normalized`` decks every factor is the identity — the deck is already
in internal units.
"""

from __future__ import annotations

from .. import _core

# SI constants for resolving the reference species of the plasma normalization.
# QE is the single source of truth for the elementary charge; it doubles as the
# eV->Joule conversion factor (1 eV = QE J) used by the CLI's thermal-speed calc.
QE = 1.602176634e-19           # elementary charge (C)
_QE = QE                       # internal alias (reference charge)
_ME = 9.1093837015e-31         # electron mass (kg)
_MP = 1.67262192369e-27        # proton mass (kg)

# reference_species -> reference mass (kg). The reference charge is always the
# positive elementary charge, so an electron (charge_C < 0) maps to q_internal = -1.
_SPECIES_MASS = {
    "electron": _ME,
    "proton": _MP,
    "ion": _MP,
    "hydrogen": _MP,
}


class Units:
    """Linear SI <-> internal-unit converter for a parsed PIC deck."""

    def __init__(self, deck) -> None:
        self._pic = _core.pic
        self._tag = self._pic.UnitTag
        if deck.units == "normalized":
            self._norm = None  # identity: deck already in internal units
        else:
            species = deck.normalization.reference_species
            m_ref = _SPECIES_MASS.get(species)
            if m_ref is None:
                raise ValueError(
                    f"normalization.reference_species {species!r} is unknown "
                    f"(expected one of {sorted(_SPECIES_MASS)})")
            self._norm = self._pic.Normalization.plasma(
                n_ref=deck.normalization.reference_density_per_m3,
                q_ref=_QE, m_ref=m_ref)

    @property
    def identity(self) -> bool:
        return self._norm is None

    @property
    def normalization(self):
        """The underlying C++ Normalization (None for an identity/normalized deck)."""
        return self._norm

    def _factor_in(self, tag) -> float:
        # Every UnitTag conversion is linear through the origin, so the factor is
        # the image of 1.0. This lets the same factor multiply scalars and arrays.
        return 1.0 if self._norm is None else self._norm.to_internal(1.0, tag)

    def _factor_out(self, tag) -> float:
        return 1.0 if self._norm is None else self._norm.to_si(1.0, tag)

    # -- SI deck -> internal solver units ------------------------------------
    def length(self, v):
        return v * self._factor_in(self._tag.length)

    def time(self, v):
        return v * self._factor_in(self._tag.time)

    def velocity(self, v):
        return v * self._factor_in(self._tag.velocity)

    def charge(self, v):
        return v * self._factor_in(self._tag.charge)

    def mass(self, v):
        return v * self._factor_in(self._tag.mass)

    def density(self, v):
        return v * self._factor_in(self._tag.density)

    # -- internal solver units -> SI (for output / diagnostics) --------------
    def length_to_si(self, v):
        return v * self._factor_out(self._tag.length)

    def time_to_si(self, v):
        return v * self._factor_out(self._tag.time)

    def velocity_to_si(self, v):
        return v * self._factor_out(self._tag.velocity)

    def e_field_to_si(self, v):
        return v * self._factor_out(self._tag.e_field)

    def b_field_to_si(self, v):
        return v * self._factor_out(self._tag.b_field)

    def field_component_to_si(self, name: str, v):
        """Scale a Yee field component to SI by its name: ex/ey/ez use the E-field
        scale, everything else (bx/by/bz, external_b*) uses the B-field scale.
        Single source of truth for the component->scale mapping."""
        return (self.e_field_to_si(v) if name in ("ex", "ey", "ez")
                else self.b_field_to_si(v))

    def external_scales(self) -> tuple[float, float, float]:
        """``(length_scale, e_field_scale, b_field_scale)`` for the external-field
        sampler: internal Yee coordinates are multiplied by ``length_scale`` to reach
        SI metres for the (SI) evaluator, and the returned SI field is divided by the
        field scales to land in internal units. All 1.0 for a normalized deck."""
        if self._norm is None:
            return (1.0, 1.0, 1.0)
        return (self._norm.length_scale(),
                self._norm.e_field_scale(),
                self._norm.b_field_scale())
