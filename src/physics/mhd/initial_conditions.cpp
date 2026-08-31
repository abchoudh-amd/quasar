#include "quasar/physics/mhd/initial_conditions.hpp"

#include "quasar/backend/memory.hpp"
#include "quasar/numerics/radial_tables.hpp"
#include "quasar/physics/mhd/kernels.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace quasar::mhd {
namespace {

// Enumerator order. The bindings and the Python deck validator read this list
// rather than mirroring it, so adding a generator is a one-line change here.
constexpr std::array<const char*, 6> kNames = {
    "brio_wu", "alfven_wave", "orszag_tang", "blast", "rotor", "confined_blob",
};

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument{std::string{message}};
}

void require_finite(Real value, const char* what) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument{std::string{"initial condition "} + what +
                                " must be finite"};
  }
}

void require_finite_state(const Real (&state)[8], const char* what) {
  static constexpr std::array<const char*, 8> kComponents = {
      "rho", "p", "vx", "vy", "vz", "bx", "by", "bz"};
  for (std::size_t c = 0; c < kComponents.size(); ++c) {
    if (!std::isfinite(state[c])) {
      throw std::invalid_argument{std::string{"initial condition "} + what +
                                  " " + kComponents[c] + " must be finite"};
    }
  }
  if (!(state[0] > Real{0})) {
    throw std::invalid_argument{std::string{"initial condition "} + what +
                                " density must be strictly positive"};
  }
  if (!(state[1] > Real{0})) {
    throw std::invalid_argument{std::string{"initial condition "} + what +
                                " pressure must be strictly positive"};
  }
}

// Kind-specific structural preconditions. These are the ones a kernel cannot
// usefully report: a bad radius pair makes every cell wrong in the same way, so
// naming the parameter here is far more diagnosable than a per-cell status bit.
void validate_kind(const MhdInitialConditionSpec& s) {
  switch (s.kind) {
    case MhdInitialConditionKind::brio_wu:
      require_finite(s.interface, "brio_wu interface");
      require_finite_state(s.left, "brio_wu left");
      require_finite_state(s.right, "brio_wu right");
      break;
    case MhdInitialConditionKind::alfven_wave:
      require_finite(s.rho, "alfven_wave rho");
      require_finite(s.pressure, "alfven_wave p");
      require_finite(s.b0, "alfven_wave b0");
      require_finite(s.total_b0, "alfven_wave reference bx");
      require_finite(s.amplitude, "alfven_wave amplitude");
      require_finite(s.magnetic_velocity_scale,
                     "alfven_wave magnetic velocity scale");
      require(s.rho > Real{0},
              "alfven_wave requires a strictly positive density");
      require(s.pressure > Real{0},
              "alfven_wave requires a strictly positive pressure");
      // Only the sign is read, and copysign(x, 0.0) is +x while the physical
      // mode is undefined: refuse rather than silently pick +x.
      require(s.total_b0 != Real{0},
              "alfven_wave requires a nonzero axial field to orient the mode");
      require(s.magnetic_velocity_scale > Real{0},
              "alfven_wave magnetic velocity scale must be positive");
      break;
    case MhdInitialConditionKind::orszag_tang:
      require_finite(s.b_uniform[0], "orszag_tang b0");
      break;
    case MhdInitialConditionKind::blast:
      require_finite(s.rho_ambient, "blast rho_ambient");
      require_finite(s.p_ambient, "blast p_ambient");
      require_finite(s.p_core, "blast p_core");
      require_finite(s.r_in, "blast r_in");
      require(s.rho_ambient > Real{0},
              "blast requires a strictly positive ambient density");
      require(s.p_ambient > Real{0} && s.p_core > Real{0},
              "blast requires strictly positive ambient and core pressures");
      break;
    case MhdInitialConditionKind::rotor:
      require_finite(s.r0, "rotor r0");
      require_finite(s.r1, "rotor r1");
      require_finite(s.u0, "rotor u0");
      require_finite(s.rho_in, "rotor rho_in");
      require_finite(s.rho_out, "rotor rho_out");
      require_finite(s.p_ambient, "rotor p");
      require(s.r0 > Real{0}, "rotor r0 must be strictly positive");
      // The taper divides by (r1 - r0) and by r inside (r0, r1); a collapsed or
      // inverted pair makes both meaningless.
      require(s.r1 > s.r0, "rotor requires r1 > r0");
      require(s.rho_in > Real{0} && s.rho_out > Real{0},
              "rotor requires strictly positive densities");
      require(s.p_ambient > Real{0},
              "rotor requires a strictly positive pressure");
      break;
    case MhdInitialConditionKind::confined_blob:
      require_finite(s.blob_half, "confined_blob blob_half");
      require_finite(s.rho_in, "confined_blob rho_in");
      require_finite(s.rho_out, "confined_blob rho_out");
      require_finite(s.p_in, "confined_blob p_in");
      require_finite(s.p_out, "confined_blob p_out");
      require_finite(s.b_uniform[2], "confined_blob bz");
      require(s.blob_half > Real{0},
              "confined_blob blob_half must be strictly positive");
      require(s.rho_in > Real{0} && s.rho_out > Real{0},
              "confined_blob requires strictly positive densities");
      require(s.p_in > Real{0} && s.p_out > Real{0},
              "confined_blob requires strictly positive pressures");
      break;
  }
}

void throw_on_seed_status(int status) {
  if ((status & kMhdSeedCoordinateNotRepresentable) != 0) {
    throw std::invalid_argument{
        "padded grid coordinates are not representable in float64"};
  }
  if ((status & kMhdSeedDensityNotPositive) != 0) {
    throw std::invalid_argument{
        "initial state must have finite, strictly positive density everywhere"};
  }
  if ((status & kMhdSeedMomentumNotRepresentable) != 0) {
    throw std::invalid_argument{"initial momentum is not representable in float64"};
  }
  if ((status & kMhdSeedCollocationNotRepresentable) != 0) {
    throw std::invalid_argument{
        "face-to-cell magnetic collocation is not representable"};
  }
  if ((status & kMhdSeedEnergyNotRepresentable) != 0) {
    throw std::invalid_argument{
        "initial state energy is not representable in float64"};
  }
  if ((status & kMhdSeedPressureNotPositive) != 0) {
    throw std::invalid_argument{
        "initial state must have strictly positive gas pressure everywhere"};
  }
}

}  // namespace

std::vector<std::string> registered_mhd_initial_conditions() {
  return std::vector<std::string>{kNames.begin(), kNames.end()};
}

MhdInitialConditionKind initial_condition_kind(std::string_view name) {
  for (std::size_t k = 0; k < kNames.size(); ++k) {
    if (name == kNames[k]) return static_cast<MhdInitialConditionKind>(k);
  }
  throw std::invalid_argument{"unknown MHD initial condition '" +
                              std::string{name} + "'"};
}

void build_initial_state(const MhdInitialConditionSpec& spec,
                         MhdField2D<Real>& out, backend::stream_t stream) {
  spec.grid.validate();
  require(std::isfinite(spec.gamma) && spec.gamma > Real{1},
          "gamma must be finite and greater than one");
  require(spec.scheme_order == 2 || spec.scheme_order == 5 ||
              spec.scheme_order == 7,
          "scheme_order must be 2, 5, or 7");
  require(std::isfinite(spec.magnetic_scale) && spec.magnetic_scale > Real{0},
          "magnetic unit scale must be finite and positive");
  require(out.grid.storage_size() == spec.grid.storage_size(),
          "output field storage does not match the initial-condition grid");
  validate_kind(spec);

  // The annular reconstruction halo must stay at positive radius: the radial
  // collocation moments below are undefined on a cell straddling the axis, and
  // an r=0 grid gets the parity closure instead (which is a property of the
  // grid, not of this seed).
  if (spec.cylindrical != 0 && spec.grid.origin_x != Real{0}) {
    const Real padded_r_lo =
        spec.grid.origin_x - static_cast<Real>(spec.grid.nghost) * spec.grid.dx();
    require(std::isfinite(padded_r_lo) && padded_r_lo > Real{0},
            "annular cylindrical geometry requires origin_x - nghost*dr > 0 so "
            "the full reconstruction halo stays at positive radius");
  }

  // Built here rather than taken from a solver: this entry point runs BEFORE a
  // solver exists (the padded state is what the solver is then seeded from),
  // which is the same reason the deck reads reconstruction_halo from the
  // registry instead of solver.grid().nghost.
  const numerics::RadialTables radial_tables =
      spec.cylindrical != 0
          ? numerics::RadialTables{spec.grid, spec.scheme_order}
          : numerics::RadialTables{};

  backend::DeviceBuffer<Real> raw_magnetic(spec.grid.storage_size());
  backend::DeviceBuffer<int> status(1);
  const int zero = 0;
  status.copy_from_host(&zero, 1);

  launch_mhd_seed_initial_state(spec, out, raw_magnetic, radial_tables.view(),
                                status.device_ptr(), stream);

  int host_status = 0;
  status.copy_to_host(&host_status, 1);
  backend::device_synchronize(stream);
  throw_on_seed_status(host_status);
}

}  // namespace quasar::mhd
