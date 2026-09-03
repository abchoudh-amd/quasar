#include "quasar/physics/pic/pic_solver.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/core/registry.hpp"

#include "quasar/physics/pic/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace quasar::numerics {

template <>
void YeeFdtd2D<2>::advance_b(YeeField2D<Real>& f, Real dt) const {
  ::launch_pic_fdtd_b_order2(f.grid, f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                             f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(), dt, nullptr);
}

template <>
void YeeFdtd2D<4>::advance_b(YeeField2D<Real>& f, Real dt) const {
  ::launch_pic_fdtd_b_order4(f.grid, f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                             f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(), dt, nullptr);
}

template <>
void YeeFdtd2D<2>::advance_e(YeeField2D<Real>& f, const JField2D<Real>& j, Real dt) const {
  ::launch_pic_fdtd_e_order2(f.grid, f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(),
                             f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                             j.jx.device_ptr(), j.jy.device_ptr(), j.jz.device_ptr(), dt, nullptr);
}

template <>
void YeeFdtd2D<4>::advance_e(YeeField2D<Real>& f, const JField2D<Real>& j, Real dt) const {
  ::launch_pic_fdtd_e_order4(f.grid, f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(),
                             f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                             j.jx.device_ptr(), j.jy.device_ptr(), j.jz.device_ptr(), dt, nullptr);
}

template <>
void BorisPusher<1>::push(pic::ParticleSpecies& s, const YeeField2D<Real>& f,
                          const YeeField2D<Real>& ext,
                          const BField2D<Real>& previous_b, Real force_dt,
                          Real position_dt, Real previous_b_weight,
                          Real current_b_weight) const {
  ::launch_pic_gather_push_shape1(f.grid, s, f, ext, previous_b, periodic_x_ ? 1 : 0,
                                  periodic_y_ ? 1 : 0, force_dt, position_dt,
                                  previous_b_weight, current_b_weight, nullptr);
  ::launch_pic_particle_error_check(s, nullptr);
}

template <>
void BorisPusher<2>::push(pic::ParticleSpecies& s, const YeeField2D<Real>& f,
                          const YeeField2D<Real>& ext,
                          const BField2D<Real>& previous_b, Real force_dt,
                          Real position_dt, Real previous_b_weight,
                          Real current_b_weight) const {
  ::launch_pic_gather_push_shape2(f.grid, s, f, ext, previous_b, periodic_x_ ? 1 : 0,
                                  periodic_y_ ? 1 : 0, force_dt, position_dt,
                                  previous_b_weight, current_b_weight, nullptr);
  ::launch_pic_particle_error_check(s, nullptr);
}

template <>
void Esirkepov2D<1>::deposit(const pic::ParticleSpecies& s, JField2D<Real>& j, Real dt) const {
  ::launch_pic_deposit_shape1(j.grid, s, j, dt, periodic_x_ ? 1 : 0,
                              periodic_y_ ? 1 : 0, nullptr);
}

template <>
void Esirkepov2D<1>::deposit_charge(const pic::ParticleSpecies& s,
                                    ScalarGrid2D<Real>& rho) const {
  ::launch_pic_charge_shape1(rho.grid, s, rho, periodic_x_ ? 1 : 0,
                             periodic_y_ ? 1 : 0, nullptr);
}

template <>
void Esirkepov2D<2>::deposit(const pic::ParticleSpecies& s, JField2D<Real>& j, Real dt) const {
  ::launch_pic_deposit_shape2(j.grid, s, j, dt, periodic_x_ ? 1 : 0,
                              periodic_y_ ? 1 : 0, nullptr);
}

template <>
void Esirkepov2D<2>::deposit_charge(const pic::ParticleSpecies& s,
                                    ScalarGrid2D<Real>& rho) const {
  ::launch_pic_charge_shape2(rho.grid, s, rho, periodic_x_ ? 1 : 0,
                             periodic_y_ ? 1 : 0, nullptr);
}

// -- Cylindrical (r,z) m=0 schemes -------------------------------------------
//
// These are the axisymmetric counterparts of YeeFdtd2D<2> / BorisPusher<S> /
// Esirkepov2D<S>. They derive the SAME IFieldSolver / IParticlePusher /
// IDepositScheme interfaces and forward to the cylindrical launch ABI. They are
// defined and registered in THIS translation unit (alongside the Cartesian
// schemes and the externally-referenced EmPic2D3V symbols) so their static
// registration initializers are not dropped by the plain (non-WHOLE_ARCHIVE)
// link — see the registration note below.  Both orders use the same
// volume-weighted radial operator; order four uses the algebraically equivalent
// D4(A)/dr+M4(A)/r form, with parity ghosts providing the regular axis closure.

template <int Order>
class YeeFdtdCyl2D final : public IFieldSolver {
 public:
  void advance_b(YeeField2D<Real>& f, Real dt) const override {
    if constexpr (Order == 4) {
      ::launch_pic_fdtd_b_cyl_order4(f.grid, f.bx.device_ptr(), f.by.device_ptr(),
                                     f.bz.device_ptr(), f.ex.device_ptr(), f.ey.device_ptr(),
                                     f.ez.device_ptr(), dt, nullptr);
    } else {
      ::launch_pic_fdtd_b_cyl_order2(f.grid, f.bx.device_ptr(), f.by.device_ptr(),
                                     f.bz.device_ptr(), f.ex.device_ptr(), f.ey.device_ptr(),
                                     f.ez.device_ptr(), dt, nullptr);
    }
  }
  void advance_e(YeeField2D<Real>& f, const JField2D<Real>& j, Real dt) const override {
    if constexpr (Order == 4) {
      ::launch_pic_fdtd_e_cyl_order4(f.grid, f.ex.device_ptr(), f.ey.device_ptr(),
                                     f.ez.device_ptr(), f.bx.device_ptr(), f.by.device_ptr(),
                                     f.bz.device_ptr(), j.jx.device_ptr(), j.jy.device_ptr(),
                                     j.jz.device_ptr(), dt, nullptr);
    } else {
      ::launch_pic_fdtd_e_cyl_order2(f.grid, f.ex.device_ptr(), f.ey.device_ptr(),
                                     f.ez.device_ptr(), f.bx.device_ptr(), f.by.device_ptr(),
                                     f.bz.device_ptr(), j.jx.device_ptr(), j.jy.device_ptr(),
                                     j.jz.device_ptr(), dt, nullptr);
    }
  }
};

template <int ShapeOrder>
class BorisCylPusher final : public IParticlePusher {
 public:
  void push(pic::ParticleSpecies& s, const YeeField2D<Real>& f, const YeeField2D<Real>& ext,
            const BField2D<Real>& previous_b, Real force_dt, Real position_dt,
            Real previous_b_weight, Real current_b_weight) const override;
  void set_periodic_axes(bool periodic_x, bool periodic_y) override {
    periodic_x_ = periodic_x;
    periodic_y_ = periodic_y;
  }

 private:
  bool periodic_x_{true};
  bool periodic_y_{true};
};

template <>
void BorisCylPusher<1>::push(pic::ParticleSpecies& s, const YeeField2D<Real>& f,
                             const YeeField2D<Real>& ext,
                             const BField2D<Real>& previous_b, Real force_dt,
                             Real position_dt, Real previous_b_weight,
                             Real current_b_weight) const {
  ::launch_pic_gather_push_cyl_shape1(f.grid, s, f, ext, previous_b, periodic_x_ ? 1 : 0,
                                      periodic_y_ ? 1 : 0, force_dt, position_dt,
                                      previous_b_weight, current_b_weight,
                                      nullptr);
  ::launch_pic_particle_error_check(s, nullptr);
}

template <>
void BorisCylPusher<2>::push(pic::ParticleSpecies& s, const YeeField2D<Real>& f,
                             const YeeField2D<Real>& ext,
                             const BField2D<Real>& previous_b, Real force_dt,
                             Real position_dt, Real previous_b_weight,
                             Real current_b_weight) const {
  ::launch_pic_gather_push_cyl_shape2(f.grid, s, f, ext, previous_b, periodic_x_ ? 1 : 0,
                                      periodic_y_ ? 1 : 0, force_dt, position_dt,
                                      previous_b_weight, current_b_weight,
                                      nullptr);
  ::launch_pic_particle_error_check(s, nullptr);
}

template <int ShapeOrder>
class EsirkepovCyl2D final : public IDepositScheme {
 public:
  void deposit(const pic::ParticleSpecies& s, JField2D<Real>& j, Real dt) const override;
  void deposit_charge(const pic::ParticleSpecies& s,
                      ScalarGrid2D<Real>& charge) const override;
  void set_periodic_axes(bool periodic_x, bool periodic_y) override {
    periodic_x_ = periodic_x;
    periodic_y_ = periodic_y;
  }

 private:
  bool periodic_x_{true};
  bool periodic_y_{true};
};

template <>
void EsirkepovCyl2D<1>::deposit(const pic::ParticleSpecies& s, JField2D<Real>& j,
                                Real dt) const {
  ::launch_pic_deposit_cyl_shape1(j.grid, s, j, dt, periodic_x_ ? 1 : 0,
                                  periodic_y_ ? 1 : 0, nullptr);
}

template <>
void EsirkepovCyl2D<1>::deposit_charge(const pic::ParticleSpecies& s,
                                       ScalarGrid2D<Real>& rho) const {
  ::launch_pic_charge_cyl_shape1(rho.grid, s, rho, periodic_x_ ? 1 : 0,
                                 periodic_y_ ? 1 : 0, nullptr);
}

template <>
void EsirkepovCyl2D<2>::deposit(const pic::ParticleSpecies& s, JField2D<Real>& j,
                                Real dt) const {
  ::launch_pic_deposit_cyl_shape2(j.grid, s, j, dt, periodic_x_ ? 1 : 0,
                                  periodic_y_ ? 1 : 0, nullptr);
}

template <>
void EsirkepovCyl2D<2>::deposit_charge(const pic::ParticleSpecies& s,
                                       ScalarGrid2D<Real>& rho) const {
  ::launch_pic_charge_cyl_shape2(rho.grid, s, rho, periodic_x_ ? 1 : 0,
                                 periodic_y_ ? 1 : 0, nullptr);
}

}  // namespace quasar::numerics

// Register the concrete field-solver / pusher / deposit schemes under deck-facing
// names so EmPic2D3V builds them through the registry instead of an if/else ladder
// over the integer order/shape. These registrations live in this TU (whose
// EmPic2D3V symbols are externally referenced), so the static initializers are
// never dropped by the linker — no WHOLE_ARCHIVE needed.
QUASAR_REGISTER_FIELD_SOLVER("yee_o2", ::quasar::numerics::YeeFdtd2D<2>)
QUASAR_REGISTER_FIELD_SOLVER("yee_o4", ::quasar::numerics::YeeFdtd2D<4>)
QUASAR_REGISTER_PUSHER("boris_cic", ::quasar::numerics::BorisPusher<1>)
QUASAR_REGISTER_PUSHER("boris_tsc", ::quasar::numerics::BorisPusher<2>)
QUASAR_REGISTER_DEPOSIT("esirkepov_cic", ::quasar::numerics::Esirkepov2D<1>)
QUASAR_REGISTER_DEPOSIT("esirkepov_tsc", ::quasar::numerics::Esirkepov2D<2>)

// Cylindrical (r,z) m=0 schemes, registered in the same TU for the same
// linker-survival reason. Additive: the Cartesian names above are untouched.
QUASAR_REGISTER_FIELD_SOLVER("yee_cyl_o2", ::quasar::numerics::YeeFdtdCyl2D<2>)
QUASAR_REGISTER_FIELD_SOLVER("yee_cyl_o4", ::quasar::numerics::YeeFdtdCyl2D<4>)
QUASAR_REGISTER_PUSHER("boris_cyl_cic", ::quasar::numerics::BorisCylPusher<1>)
QUASAR_REGISTER_PUSHER("boris_cyl_tsc", ::quasar::numerics::BorisCylPusher<2>)
QUASAR_REGISTER_DEPOSIT("esirkepov_cyl_cic", ::quasar::numerics::EsirkepovCyl2D<1>)
QUASAR_REGISTER_DEPOSIT("esirkepov_cyl_tsc", ::quasar::numerics::EsirkepovCyl2D<2>)

namespace quasar::pic {

namespace {

struct ScaledLongValue {
  long double mantissa{0.0L};
  int exponent{0};
};

ScaledLongValue scaled_long_product(
    std::initializer_list<long double> factors) {
  long double mantissa = 1.0L;
  int exponent = 0;
  for (const long double factor : factors) {
    if (factor == 0.0L) return {};
    int factor_exponent = 0;
    const long double factor_mantissa = std::frexp(factor, &factor_exponent);
    mantissa *= factor_mantissa;
    exponent += factor_exponent;
    int adjustment = 0;
    mantissa = std::frexp(mantissa, &adjustment);
    exponent += adjustment;
  }
  return {mantissa, exponent};
}

class ScaledCompensatedSum {
 public:
  void add(ScaledLongValue value) {
    if (value.mantissa == 0.0L) return;
    if (!initialized_) {
      exponent_ = value.exponent;
      initialized_ = true;
    } else if (value.exponent > exponent_) {
      const int shift = exponent_ - value.exponent;
      sum_ = std::scalbn(sum_, shift);
      correction_ = std::scalbn(correction_, shift);
      exponent_ = value.exponent;
    }
    const long double term = std::scalbn(
        value.mantissa, value.exponent - exponent_);
    const long double next = sum_ + term;
    if (std::fabs(sum_) >= std::fabs(term)) {
      correction_ += (sum_ - next) + term;
    } else {
      correction_ += (term - next) + sum_;
    }
    sum_ = next;
  }

  ScaledLongValue normalized() const {
    if (!initialized_) return {};
    const long double total = sum_ + correction_;
    if (total == 0.0L) return {};
    int adjustment = 0;
    const long double mantissa = std::frexp(total, &adjustment);
    return {mantissa, exponent_ + adjustment};
  }

 private:
  bool initialized_{false};
  int exponent_{0};
  long double sum_{0.0L};
  long double correction_{0.0L};
};

// Bring a device reduction result into the host accumulator's frame. The
// kernel's sum and its Kahan correction are added in long double so the
// compensation it accumulated is not thrown away at the boundary.
ScaledLongValue to_scaled_long(const PicScaledSum& sum) {
  if (!sum.initialized) return {};
  const long double total = static_cast<long double>(sum.sum)
                          + static_cast<long double>(sum.correction);
  if (total == 0.0L) return {};
  int adjustment = 0;
  return {std::frexp(total, &adjustment), sum.exponent + adjustment};
}

bool is_cylindrical(const std::string& geometry) { return geometry == "cylindrical"; }

int current_ghost_mode(const boundary::IFieldBoundary& boundary,
                       std::string_view boundary_name) {
  const int mode = boundary.ghost_continuation_mode();
  if (mode >= 0 && mode <= 4) return mode;
  throw std::invalid_argument{
      "EmPic2D3V: fourth-order current correction does not know the field "
      "ghost continuation for boundary '" + std::string{boundary_name} + "'"};
}

// The full set of geometry-dependent choices the driver makes: the resolved
// field-solver / pusher / deposit registry names and whether the geometry needs
// the on-axis (x_lo, r=0) boundary override. Resolving all of them in one
// function means a new geometry is a single branch added here — the driver below
// never branches on the geometry string itself, so it stays free of the
// per-geometry if/else ladder (the plugin-selection convention in CLAUDE.md).
struct SchemeFamily {
  bool on_axis_x_lo{false};
  std::string field_solver{};
  std::string pusher{};
  std::string deposit{};
};

// Maps the deck's geometry/order/shape vocabulary to the registry names
// registered just above. Kept adjacent to the QUASAR_REGISTER_* lines so a new
// scheme's name appears in exactly one place: register it, then add its
// vocabulary here. A cylindrical run resolves to the "_cyl_" family and flags
// the r=0 axis BC only when the radial origin is zero; a
// cartesian run keeps the existing names verbatim and needs no axis override.
SchemeFamily resolve_scheme_family(const std::string& geometry, int fdtd_order,
                                   const std::string& shape, Real radial_origin) {
  if (is_cylindrical(geometry)) {
    return SchemeFamily{/*on_axis_x_lo=*/radial_origin == Real{0},
                        "yee_cyl_o" + std::to_string(fdtd_order),
                        "boris_cyl_" + shape, "esirkepov_cyl_" + shape};
  }
  return SchemeFamily{/*on_axis_x_lo=*/false, "yee_o" + std::to_string(fdtd_order),
                      "boris_" + shape, "esirkepov_" + shape};
}

template <class Base>
void require_registered(std::string_view name, const char* what) {
  if (name.empty() || !Registry<Base>::instance().contains(name)) {
    throw std::invalid_argument{std::string{"EmPic2D3V: unknown "} + what + " '"
                                + std::string{name} + "'"};
  }
}

void require_periodic_pair(const std::array<std::string, 4>& sides,
                           int lo, int hi, const char* what, const char* axis) {
  const bool lo_periodic = sides[lo] == "periodic";
  const bool hi_periodic = sides[hi] == "periodic";
  if (lo_periodic != hi_periodic) {
    throw std::invalid_argument{
        std::string{"EmPic2D3V: "} + what + " periodic boundaries on the "
        + axis + " axis must be specified on both sides"};
  }
}

void validate_normalization(const Normalization& normalization) {
  if (normalization.is_identity()) return;

  if (!(std::isfinite(normalization.n_ref) && normalization.n_ref > Real{0})
      || !(std::isfinite(normalization.m_ref) && normalization.m_ref > Real{0})
      || !(std::isfinite(normalization.omega_p_ref)
           && normalization.omega_p_ref > Real{0})
      || !(std::isfinite(normalization.q_ref)
           && normalization.q_ref != Real{0})) {
    throw std::invalid_argument{
        "EmPic2D3V: normalization must be identity or have finite positive "
        "n_ref, m_ref, and omega_p_ref and a finite nonzero q_ref"};
  }

  const std::array<Real, 5> scales{
      normalization.length_scale(), normalization.time_scale(),
      std::abs(normalization.e_field_scale()),
      std::abs(normalization.b_field_scale()),
      normalization.temperature_eV_scale()};
  if (!std::all_of(scales.begin(), scales.end(), [](Real scale) {
        return std::isfinite(scale) && scale > Real{0};
      })) {
    throw std::invalid_argument{
        "EmPic2D3V: normalization conversion scales must be finite and nonzero"};
  }

  // Normalization remains a public aggregate for binding compatibility, so a
  // direct C++ caller can manually write an omega inconsistent with n/q/m.
  // Reconstruct it through the authoritative scale-safe factory and compare as
  // a bounded ratio (rather than subtracting two potentially huge values).
  const Normalization expected = Normalization::plasma(
      normalization.n_ref, normalization.q_ref, normalization.m_ref);
  const Real omega_hi = std::max(normalization.omega_p_ref,
                                 expected.omega_p_ref);
  const Real omega_lo = std::min(normalization.omega_p_ref,
                                 expected.omega_p_ref);
  const Real relative_error = Real{1} - omega_lo / omega_hi;
  constexpr Real tolerance = Real{64} * std::numeric_limits<Real>::epsilon();
  if (!(relative_error <= tolerance)) {
    throw std::invalid_argument{
        "EmPic2D3V: normalization omega_p_ref is inconsistent with n_ref, "
        "q_ref, and m_ref"};
  }
}

// Validate every public C++ configuration surface before constructing a single
// grid-sized device buffer.  The Python deck performs equivalent checks, but
// EmPic2D3V is also a direct C++ API and cannot rely on that front end.
EmPicConfig validate_config(EmPicConfig cfg) {
  cfg.grid.validate();
  if (cfg.geometry != "cartesian" && cfg.geometry != "cylindrical") {
    throw std::invalid_argument{
        "EmPic2D3V: geometry must be 'cartesian' or 'cylindrical'"};
  }
  if (cfg.plane != "xy" && cfg.plane != "xz") {
    throw std::invalid_argument{"EmPic2D3V: plane must be 'xy' or 'xz'"};
  }
  if (cfg.fdtd_order != 2 && cfg.fdtd_order != 4) {
    throw std::invalid_argument{"EmPic2D3V: fdtd_order must be 2 or 4"};
  }
  // A fourth-order non-periodic face needs two distinct source layers for its
  // centred ghost continuation. With one physical cell the low and high ghost
  // fills overlap and an active stencil can consume a not-yet-defined ghost.
  // Apply the same minimum uniformly across boundary topologies so a later BC
  // change cannot turn an accepted grid into an ill-defined scheme.
  if (cfg.fdtd_order == 4 && (cfg.grid.nx < 2 || cfg.grid.ny < 2)) {
    throw std::invalid_argument{
        "EmPic2D3V: fourth-order FDTD requires at least two cells in each "
        "dimension"};
  }
  if (cfg.shape != "cic" && cfg.shape != "tsc") {
    throw std::invalid_argument{"EmPic2D3V: shape must be 'cic' or 'tsc'"};
  }

  // One high-halo slot stores the physical Yee high face. TSC gathering needs
  // the following slot as its first true ghost even with an order-two curl.
  const int needed_halo = std::max(required_nghost(cfg.fdtd_order),
                                   cfg.shape == "tsc" ? 2 : 1);
  if (cfg.grid.nghost < needed_halo) {
    throw std::invalid_argument{
        "EmPic2D3V: grid nghost is too small for the selected FDTD order and "
        "particle shape"};
  }

  const bool cylindrical = is_cylindrical(cfg.geometry);
  if (cylindrical && cfg.grid.origin_x < Real{0}) {
    throw std::invalid_argument{
        "EmPic2D3V: cylindrical radial origin must be non-negative"};
  }

  // r=0 is a coordinate axis, not a deck-selected wall. Keep the established
  // convenience that a default x_lo entry is replaced by the regularity BC.
  // Doing this before topology validation prevents that default placeholder
  // from looking like a one-sided periodic radial boundary.
  if (cylindrical && cfg.grid.origin_x == Real{0}) {
    constexpr int x_lo = static_cast<int>(Side::x_lo);
    if (cfg.boundary.field[x_lo] != "periodic"
        && cfg.boundary.field[x_lo] != "axis") {
      throw std::invalid_argument{
          "EmPic2D3V: cylindrical r=0 field boundary must be 'axis' (the "
          "default 'periodic' placeholder is auto-replaced)"};
    }
    if (cfg.boundary.particle[x_lo] != "periodic"
        && cfg.boundary.particle[x_lo] != "axis") {
      throw std::invalid_argument{
          "EmPic2D3V: cylindrical r=0 particle boundary must be 'axis' (the "
          "default 'periodic' placeholder is auto-replaced)"};
    }
    cfg.boundary.field[x_lo] = "axis";
    cfg.boundary.particle[x_lo] = "axis";
  }

  std::array<bool, 4> field_internal{};
  std::array<bool, 4> particle_internal{};
  for (int side = 0; side < 4; ++side) {
    const bool axis_requested = cfg.boundary.field[side] == "axis"
                             || cfg.boundary.particle[side] == "axis";
    if (axis_requested
        && (!cylindrical || cfg.grid.origin_x != Real{0}
            || side != static_cast<int>(Side::x_lo))) {
      throw std::invalid_argument{
          "EmPic2D3V: the 'axis' boundary is valid only at cylindrical r=0 "
          "(x_lo)"};
    }
    require_registered<boundary::IFieldBoundary>(cfg.boundary.field[side],
                                                  "field boundary");
    require_registered<boundary::IParticleBoundary>(cfg.boundary.particle[side],
                                                     "particle boundary");
    auto field_boundary =
        Registry<boundary::IFieldBoundary>::instance().create(
            cfg.boundary.field[side]);
    auto particle_boundary =
        Registry<boundary::IParticleBoundary>::instance().create(
            cfg.boundary.particle[side]);
    field_internal[side] = field_boundary->is_internal_cut();
    particle_internal[side] = particle_boundary->is_internal_cut();
    if (cfg.fdtd_order == 4) {
      // Fail before allocating fields if a future plugin does not declare one
      // of the compact-current ghost continuations understood by this solver.
      (void)current_ghost_mode(*field_boundary, cfg.boundary.field[side]);
    }
  }

  require_periodic_pair(cfg.boundary.field, 0, 1, "field", "x/r");
  require_periodic_pair(cfg.boundary.field, 2, 3, "field", "y/z");
  require_periodic_pair(cfg.boundary.particle, 0, 1, "particle", "x/r");
  require_periodic_pair(cfg.boundary.particle, 2, 3, "particle", "y/z");

  for (int side = 0; side < 4; ++side) {
    if (field_internal[side] != particle_internal[side]) {
      throw std::invalid_argument{
          "EmPic2D3V: field and particle internal topology must match on every side"};
    }
  }
  const bool field_periodic_x = cfg.boundary.field[0] == "periodic";
  const bool field_periodic_y = cfg.boundary.field[2] == "periodic";
  const bool particle_periodic_x = cfg.boundary.particle[0] == "periodic";
  const bool particle_periodic_y = cfg.boundary.particle[2] == "periodic";
  if (field_periodic_x != particle_periodic_x
      || field_periodic_y != particle_periodic_y) {
    throw std::invalid_argument{
        "EmPic2D3V: field and particle periodicity must match on each axis; "
        "a periodic Maxwell domain cannot exchange charge through a particle "
        "wall"};
  }

  // A relativistic-speed guard and the Maxwell CFL bound limit a particle to
  // less than one cell per position update.  At a periodic seam the boundary
  // kernel shifts both the new and previous positions by one domain length so
  // charge-conserving deposition sees that short trajectory.  The shifted
  // previous position can therefore lie as far as one cell outside the
  // opposite face.  Reject translated domains for which either image bound is
  // not representable; otherwise an otherwise finite seam crossing can create
  // an infinity only when the boundary kernel applies the domain shift.
  const auto validate_periodic_images = [](bool periodic, Real origin,
                                           Real spacing, int cells,
                                           const char* axis) {
    if (!periodic) return;
    const Real low_image = std::fma(Real{-1}, spacing, origin);
    const Real high_image = std::fma(static_cast<Real>(cells + 1),
                                     spacing, origin);
    if (!(std::isfinite(low_image) && std::isfinite(high_image))) {
      throw std::overflow_error{
          std::string{"EmPic2D3V: periodic particle images on the "} + axis
          + " axis are not representable"};
    }
  };
  validate_periodic_images(particle_periodic_x, cfg.grid.origin_x,
                           cfg.grid.dx(), cfg.grid.nx, "x/r");
  validate_periodic_images(particle_periodic_y, cfg.grid.origin_y,
                           cfg.grid.dy(), cfg.grid.ny, "y/z");

  // Before the final absorbing-wall deposit, a crossed particle centre is
  // moved just beyond the point where its complete B-spline support vanishes:
  // half a cell for CIC and one cell for TSC.  Preflight the exact affine
  // expressions used by the device kernel so that boundary loss cannot turn a
  // finite particle coordinate into an infinity.
  const Real support_radius = cfg.shape == "tsc" ? Real{1} : Real{0.5};
  const auto validate_absorbing_endpoint = [&](int side, Real origin,
                                                Real length, Real spacing,
                                                bool high,
                                                const char* axis) {
    if (cfg.boundary.particle[side] != "absorbing") return;
    const Real wall = high ? origin + length : origin;
    const Real direction = high ? support_radius : -support_radius;
    const Real endpoint = std::fma(direction, spacing, wall);
    if (!std::isfinite(endpoint)) {
      throw std::overflow_error{
          std::string{"EmPic2D3V: absorbing deposition endpoint on the "}
          + axis + (high ? " high" : " low")
          + " side is not representable"};
    }
  };
  validate_absorbing_endpoint(0, cfg.grid.origin_x, cfg.grid.lx,
                              cfg.grid.dx(), false, "x/r");
  validate_absorbing_endpoint(1, cfg.grid.origin_x, cfg.grid.lx,
                              cfg.grid.dx(), true, "x/r");
  validate_absorbing_endpoint(2, cfg.grid.origin_y, cfg.grid.ly,
                              cfg.grid.dy(), false, "y/z");
  validate_absorbing_endpoint(3, cfg.grid.origin_y, cfg.grid.ly,
                              cfg.grid.dy(), true, "y/z");

  if (cylindrical) {
    if (cfg.boundary.field[1] == "periodic"
        || cfg.boundary.particle[1] == "periodic") {
      throw std::invalid_argument{
          "EmPic2D3V: cylindrical outer-radius boundaries must be non-periodic"};
    }
    if (cfg.grid.origin_x > Real{0}) {
      if (cfg.boundary.field[0] == "periodic"
          || cfg.boundary.particle[0] == "periodic") {
        throw std::invalid_argument{
            "EmPic2D3V: annular inner-radius boundaries must be non-periodic"};
      }
      const Real padded_r_lo = cfg.grid.origin_x
          - static_cast<Real>(cfg.grid.nghost) * cfg.grid.dx();
      const bool internal_radial_low =
          field_internal[0] && particle_internal[0];
      const bool valid = std::isfinite(padded_r_lo) &&
          (internal_radial_low ? padded_r_lo >= Real{0}
                               : padded_r_lo > Real{0});
      if (!valid) {
        throw std::invalid_argument{
            internal_radial_low
                ? "EmPic2D3V: an internal annular tile requires "
                  "origin_x - nghost*dr >= 0 so its radial halo does not "
                  "cross the global axis"
                : "EmPic2D3V: annular geometry requires "
                  "origin_x - nghost*dr > 0 so every radial ghost remains "
                  "at positive radius"};
      }
    }
    // Order-four curls and external-field sampling evaluate true radial ghosts,
    // not only the finite physical upper bound checked by Grid2D. Match the
    // largest padded cell-centre coordinate used by those paths and reject an
    // overflow before allocating any field buffer.
    const Real padded_r_hi = cfg.grid.origin_x
        + (static_cast<Real>(cfg.grid.nx + cfg.grid.nghost) - Real{0.5})
              * cfg.grid.dx();
    if (!std::isfinite(padded_r_hi)) {
      throw std::overflow_error{
          "EmPic2D3V: padded cylindrical high-radius coordinate is not "
          "representable"};
    }
  }

  const bool x_outflow = cfg.boundary.field[0] == "outflow"
                      || cfg.boundary.field[1] == "outflow";
  const bool y_outflow = cfg.boundary.field[2] == "outflow"
                      || cfg.boundary.field[3] == "outflow";
  if (x_outflow && cfg.grid.nx < 2) {
    throw std::invalid_argument{
        "EmPic2D3V: an x/r outflow boundary requires at least two cells"};
  }
  if (y_outflow && cfg.grid.ny < 2) {
    throw std::invalid_argument{
        "EmPic2D3V: a y/z outflow boundary requires at least two cells"};
  }

  const auto family = resolve_scheme_family(cfg.geometry, cfg.fdtd_order,
                                             cfg.shape, cfg.grid.origin_x);
  require_registered<numerics::IFieldSolver>(family.field_solver,
                                              "field solver");
  require_registered<numerics::IParticlePusher>(family.pusher,
                                                 "particle pusher");
  require_registered<numerics::IDepositScheme>(family.deposit,
                                                "deposit scheme");
  for (const auto& spec : cfg.filters) {
    if (spec.passes < 1) {
      throw std::invalid_argument{
          "EmPic2D3V: current-filter passes must be positive"};
    }
    require_registered<numerics::ICurrentFilter>(spec.name, "current filter");
  }
  validate_normalization(cfg.normalization);
  return cfg;
}

}  // namespace

EmPic2D3V::EmPic2D3V(EmPicConfig cfg)
  : cfg_{validate_config(std::move(cfg))},
    grid_{cfg_.grid},
    fields_{grid_},
    external_fields_{grid_},
    previous_b_{grid_},
    current_{grid_},
    charge_{grid_},
    next_charge_{grid_} {
  // Resolve every geometry-dependent choice (scheme names + axis BC) in one
  // place so the driver below never branches on the geometry string itself.
  const auto family = resolve_scheme_family(cfg_.geometry, cfg_.fdtd_order,
                                            cfg_.shape, grid_.origin_x);
  // In cylindrical (r,z) mode validate_config replaced only the exact default
  // periodic x_lo placeholder with the on-axis condition. Explicit walls at
  // r=0 were rejected rather than silently changing the requested physics.
  (void)family.on_axis_x_lo;  // Axis override was applied pre-allocation.
  // Build per-side boundary conditions through the registry (the documented
  // pluggable construction path); concrete BCs self-register in src/boundary.
  // The FDTD stencil is ghost-aware, so these field BCs run every step (see step).
  for (int side = 0; side < 4; ++side) {
    particle_bcs_[side] = Registry<boundary::IParticleBoundary>::instance().create(
        cfg_.boundary.particle[side]);
    field_bcs_[side] = Registry<boundary::IFieldBoundary>::instance().create(
        cfg_.boundary.field[side]);
    // The FDTD order fixes how many boundary layers a one-sided/characteristic
    // closure must rewrite; let each field BC pick its order-dependent kernel.
    field_bcs_[side]->configure(cfg_.fdtd_order);
    // PEC/outflow choose between Cartesian and cylindrical component/parity
    // kernels for every radial domain, including annuli whose x_lo is not the
    // symmetry axis.  `on_axis_x_lo` is only the boundary override decision.
    field_bcs_[side]->configure_geometry(is_cylindrical(cfg_.geometry));
    particle_bcs_[side]->configure_geometry(is_cylindrical(cfg_.geometry));
  }
  // A doubly-outflow corner is owned by one diagonal characteristic. Both face
  // Mur kernels skip only that shared doubly-tangential component; PEC parity and
  // mixed PEC/outflow closures act on different physical/ghost degrees of freedom
  // and therefore need no skip.
  const auto& fb = cfg_.boundary.field;
  const bool x_lo_out = fb[0] == "outflow";
  const bool x_hi_out = fb[1] == "outflow";
  const bool y_lo_out = fb[2] == "outflow";
  const bool y_hi_out = fb[3] == "outflow";
  field_bcs_[0]->set_corner_skip(y_lo_out, y_hi_out);
  field_bcs_[1]->set_corner_skip(y_lo_out, y_hi_out);
  field_bcs_[2]->set_corner_skip(x_lo_out, x_hi_out);
  field_bcs_[3]->set_corner_skip(x_lo_out, x_hi_out);
  if (x_lo_out && y_lo_out) outflow_corner_mask_ |= 1u << 0;
  if (x_hi_out && y_lo_out) outflow_corner_mask_ |= 1u << 1;
  if (x_lo_out && y_hi_out) outflow_corner_mask_ |= 1u << 2;
  if (x_hi_out && y_hi_out) outflow_corner_mask_ |= 1u << 3;
  if (outflow_corner_mask_ != 0u) {
    outflow_corner_history_ = backend::DeviceBuffer<Real>{8};
  }
  // Build the order/shape-templated schemes through the registry (same pluggable
  // path as the BCs/filters). The vocabulary->name mapping lives next to the
  // registrations above, so the driver passes a resolved name through verbatim;
  // Registry::create throws on an unregistered name.
  field_solver_ = Registry<numerics::IFieldSolver>::instance().create(family.field_solver);
  pusher_ = Registry<numerics::IParticlePusher>::instance().create(family.pusher);
  deposit_ = Registry<numerics::IDepositScheme>::instance().create(family.deposit);
  // Validation guarantees the field and particle periodic topology agrees on
  // each axis. A periodic Maxwell divergence telescopes to zero net charge, so
  // letting particles cross a wall on that same axis would violate Gauss's law.
  // Keep the two values separate here because they are passed to different
  // registry interfaces, not because mismatched topology is supported.
  const auto& pb = cfg_.boundary.particle;
  periodic_x_ = pb[0] == "periodic" && pb[1] == "periodic";
  periodic_y_ = pb[2] == "periodic" && pb[3] == "periodic";
  deposit_->set_periodic_axes(periodic_x_, periodic_y_);
  const bool field_periodic_x = fb[0] == "periodic" && fb[1] == "periodic";
  const bool field_periodic_y = fb[2] == "periodic" && fb[3] == "periodic";
  pusher_->set_periodic_axes(field_periodic_x, field_periodic_y);

  if (cfg_.fdtd_order == 4) {
    const std::size_t n = grid_.storage_size();
    current_rhs_x_ = backend::DeviceBuffer<Real>{n};
    current_rhs_y_ = backend::DeviceBuffer<Real>{n};
    current_iter_x_ = backend::DeviceBuffer<Real>{n};
    current_iter_y_ = backend::DeviceBuffer<Real>{n};
  }

  // Build the current-smoothing pipeline from the deck via the registry (same
  // pluggable path as the BCs); concrete filters self-register in src/numerics.
  for (const auto& spec : cfg_.filters) {
    auto filter = Registry<numerics::ICurrentFilter>::instance().create(spec.name);
    filter->set_passes(spec.passes);
    filters_.add(std::move(filter));
  }
}

void EmPic2D3V::add_species(ParticleSpecies s) {
  if (evolution_started_) {
    throw std::logic_error{
        "EmPic2D3V::add_species: species cannot be added after evolution begins"};
  }
  const bool has_outflow = std::any_of(
      cfg_.boundary.field.begin(), cfg_.boundary.field.end(),
      [](const auto& name) { return name == "outflow"; });
  if (s.charge() != Real{0} && has_outflow) {
    throw std::invalid_argument{
        "EmPic2D3V::add_species: charged particles are not supported with "
        "Mur outflow field boundaries because the boundary correction is not "
        "charge/current compatible; use periodic or PEC field boundaries"};
  }
  if (std::any_of(species_.begin(), species_.end(), [&](const auto& existing) {
        return existing.name() == s.name();
      })) {
    throw std::invalid_argument{
        "EmPic2D3V::add_species: species names must be unique"};
  }
  // Initial charge is deposited before the first push/boundary pass. Reject a
  // direct-C++ species whose live centres start outside the physical domain;
  // otherwise its shape tail would be silently clipped or folded according to
  // a boundary it has not actually crossed. Boundary points themselves are
  // admissible (periodic high faces alias low; wall faces are physical).
  //
  // Read on device. This used to call s.to_host(), pulling ALL ELEVEN particle
  // planes across the bus to look at two of them and a flag -- and the species
  // is already resident, so there was nothing to gain by it.
  const Real x_lo = grid_.origin_x;
  const Real x_hi = grid_.origin_x + grid_.lx;
  const Real y_lo = grid_.origin_y;
  const Real y_hi = grid_.origin_y + grid_.ly;
  if (s.size() > 0) {
    backend::DeviceBuffer<int> status{1};
    const int zero = 0;
    status.copy_from_host(&zero, 1);
    launch_pic_check_initial_domain(
        static_cast<std::uint64_t>(s.size()), s.x(), s.y(), s.alive(), x_lo,
        x_hi, y_lo, y_hi, status.device_ptr(), nullptr);
    int host_status = 0;
    status.copy_to_host(&host_status, 1);
    backend::device_synchronize(nullptr);
    if (host_status != 0) {
      throw std::invalid_argument{
          "EmPic2D3V::add_species: every live initial particle must lie "
          "inside the physical domain"};
    }
  }
  s.set_grid(grid_);
  species_.push_back(std::move(s));
  charge_valid_ = false;
  background_initialized_ = false;
  background_charge_density_ = Real{0};
}

void EmPic2D3V::set_species_particles(
    std::size_t index, const std::vector<Real>& x,
    const std::vector<Real>& y, const std::vector<Real>& vx,
    const std::vector<Real>& vy, const std::vector<Real>& vz,
    const std::vector<Real>& weight) {
  if (evolution_started_) {
    throw std::logic_error{
        "EmPic2D3V::set_species_particles: particles cannot be replaced after "
        "evolution begins"};
  }
  if (index >= species_.size()) {
    throw std::out_of_range{
        "EmPic2D3V::set_species_particles: species index out of range"};
  }
  if (x.size() != y.size()) {
    throw std::invalid_argument{
        "EmPic2D3V::set_species_particles: x/y size mismatch"};
  }
  const Real x_lo = grid_.origin_x;
  const Real x_hi = grid_.origin_x + grid_.lx;
  const Real y_lo = grid_.origin_y;
  const Real y_hi = grid_.origin_y + grid_.ly;
  for (std::size_t p = 0; p < x.size(); ++p) {
    if (!(std::isfinite(x[p]) && std::isfinite(y[p])
          && x[p] >= x_lo && x[p] <= x_hi
          && y[p] >= y_lo && y[p] <= y_hi)) {
      throw std::invalid_argument{
          "EmPic2D3V::set_species_particles: every initial particle must lie "
          "inside the physical domain"};
    }
  }
  // ParticleSpecies validates all remaining array sizes/values before copying,
  // so a failed upload leaves the existing device state unchanged.
  species_[index].set_host_particles(x, y, vx, vy, vz, weight);
  charge_valid_ = false;
  background_initialized_ = false;
  background_charge_density_ = Real{0};
}

void EmPic2D3V::sample_species_particles(std::size_t index,
                                         ParticleSampleConfig config) {
  if (evolution_started_) {
    throw std::logic_error{
        "EmPic2D3V::sample_species_particles: particles cannot be replaced "
        "after evolution begins"};
  }
  if (index >= species_.size()) {
    throw std::out_of_range{
        "EmPic2D3V::sample_species_particles: species index out of range"};
  }
  // The domain is the solver's, not the deck's: the sampler checks every
  // coordinate it produces against it, which is the device counterpart of the
  // host loop set_species_particles runs over an uploaded array.
  config.domain_origin_x = grid_.origin_x;
  config.domain_origin_y = grid_.origin_y;
  config.domain_lx = grid_.lx;
  config.domain_ly = grid_.ly;
  // sample_species publishes the count only after the status word comes back
  // clean, so a rejected configuration leaves the existing state unchanged.
  sample_species(species_[index], config, nullptr);
  charge_valid_ = false;
  background_initialized_ = false;
  background_charge_density_ = Real{0};
}

void EmPic2D3V::initialize_neutralizing_background() {
  if (background_initialized_) return;
  background_charge_density_ = Real{0};
  const bool periodic_torus = periodic_x_ && periodic_y_;
  ScaledLongValue total_charge{};
  if (cfg_.neutralizing_background || periodic_torus) {
    // Per-species device reduction, combined across species on the host. The
    // O(particles) work is on the GPU; only the handful of per-species
    // subtotals are folded here, which is why this stays a host loop.
    ScaledCompensatedSum particle_charge;
    ScaledCompensatedSum absolute_particle_charge;
    for (const auto& s : species_) {
      const PicChargeTotals totals = launch_pic_total_charge(s, nullptr);
      particle_charge.add(to_scaled_long(totals.net));
      absolute_particle_charge.add(to_scaled_long(totals.absolute));
    }
    total_charge = particle_charge.normalized();

    if (periodic_torus && !cfg_.neutralizing_background
        && total_charge.mantissa != 0.0L) {
      const ScaledLongValue absolute_charge =
          absolute_particle_charge.normalized();
      const long double relative_imbalance = std::scalbn(
          std::fabs(total_charge.mantissa) / absolute_charge.mantissa,
          total_charge.exponent - absolute_charge.exponent);
      // Charge products originate as Real values.  Treat an imbalance below a
      // modest multiple of their working-precision roundoff as numerically zero;
      // any resolved remainder is incompatible with integral div(E)=0 on a
      // doubly periodic domain.
      constexpr long double neutrality_tolerance =
          64.0L * static_cast<long double>(
              std::numeric_limits<Real>::epsilon());
      if (!(relative_imbalance <= neutrality_tolerance)) {
        throw std::invalid_argument{
            "EmPic2D3V: a doubly periodic field domain requires zero net "
            "initial particle charge; add a neutralizing species or set "
            "neutralizing_background=true"};
      }
    }
  }

  if (cfg_.neutralizing_background) {
    ScaledLongValue volume = scaled_long_product({
        static_cast<long double>(grid_.lx),
        static_cast<long double>(grid_.ly)});
    if (is_cylindrical(cfg_.geometry)) {
      // pi*((r0+L)^2-r0^2)*Z, factored to avoid catastrophic cancellation
      // for a thin annulus at large radius.
      const long double radial_mid = static_cast<long double>(grid_.origin_x)
                                   + 0.5L * static_cast<long double>(grid_.lx);
      volume = scaled_long_product({
          2.0L, static_cast<long double>(pi_v<Real>),
          static_cast<long double>(grid_.lx), radial_mid,
          static_cast<long double>(grid_.ly)});
    }
    if (!(std::isfinite(volume.mantissa) && volume.mantissa > 0.0L)) {
      throw std::overflow_error{
          "EmPic2D3V: domain volume is not representable"};
    }
    long double density = 0.0L;
    if (total_charge.mantissa != 0.0L) {
      long double ratio = -total_charge.mantissa / volume.mantissa;
      int adjustment = 0;
      ratio = std::frexp(ratio, &adjustment);
      density = std::scalbn(
          ratio, total_charge.exponent - volume.exponent + adjustment);
    }
    const long double real_max =
        static_cast<long double>(std::numeric_limits<Real>::max());
    if (!std::isfinite(density) || std::fabs(density) > real_max) {
      throw std::overflow_error{
          "EmPic2D3V: neutralizing background density is not representable"};
    }
    background_charge_density_ = static_cast<Real>(density);
    if (density != 0.0L && background_charge_density_ == Real{0}) {
      throw std::underflow_error{
          "EmPic2D3V: neutralizing background density underflows Real"};
    }
  }
  background_initialized_ = true;
}

void EmPic2D3V::deposit_charge_density(ScalarGrid2D<Real>& charge) {
  backend::device_memset_async(charge.values.device_ptr(), 0, charge.values.bytes(), nullptr);
  for (const auto& s : species_) {
    deposit_->deposit_charge(s, charge);
  }
  // Reflecting walls mirror finite-shape charge tails from their ghost cells
  // into the physical domain.  Applying x sides before y sides also folds a
  // doubly-reflected corner tail exactly once along each axis.
  for (int side = 0; side < 4; ++side) {
    particle_bcs_[side]->fold_charge(charge, static_cast<Side>(side));
  }
  ::launch_pic_add_uniform_charge(grid_, charge, background_charge_density_, nullptr);
}

void EmPic2D3V::ensure_charge_density() {
  if (charge_valid_) return;
  initialize_neutralizing_background();
  deposit_charge_density(charge_);
  // Charge deposition uses the same sticky range flag as current deposition.
  // Drain it here so a diagnostic or the first step never accepts a partially
  // deposited initial charge field.
  check_deposit_overflow();
  // Foldback and uniform-background addition run after the atomic deposit and
  // can overflow independently of its per-species flag. Do not publish charge
  // until the fully transformed allocation, including ghosts, is finite.
  ::launch_pic_validate_finite_sources(
      grid_, nullptr, &charge_, source_finite_error_.device_ptr(), nullptr);
  charge_valid_ = true;
}

const ScalarGrid2D<Real>& EmPic2D3V::charge_density() {
  ensure_charge_density();
  return charge_;
}

const ScalarGrid2D<Real>& EmPic2D3V::charge_density() const {
  if (!charge_valid_) {
    throw std::logic_error{
        "EmPic2D3V::charge_density: charge has not been materialized; use a "
        "non-const solver for first access"};
  }
  return charge_;
}

bool EmPic2D3V::has_absorbing_boundary() const noexcept {
  for (int side = 0; side < 4; ++side) {
    if (cfg_.boundary.particle[side] == "absorbing") {
      return true;
    }
  }
  return false;
}

void EmPic2D3V::seed_initial_fields(const PicInitialFieldSpec& spec, Real dt) {
  if (evolution_started_) {
    throw std::logic_error{
        "EmPic2D3V::seed_initial_fields: fields cannot be seeded after "
        "evolution begins"};
  }
  PicInitialFieldSpec resolved = spec;
  // The solver's grid is authoritative: the ctor may have widened the halo
  // beyond what the deck asked for (order-two TSC needs more than the curl
  // does), and the seed has to be laid out on the halo the solver actually has.
  resolved.grid = grid_;

  backend::DeviceBuffer<int> status{1};
  const int zero = 0;
  status.copy_from_host(&zero, 1);

  launch_pic_seed_initial_fields(resolved, fields_, status.device_ptr(),
                                 nullptr);

  const bool cylindrical_standing_mode =
      resolved.cylindrical != 0
      && resolved.kind == PicInitialFieldKind::seed_perturbation;
  if (cylindrical_standing_mode) {
    if (!(std::isfinite(dt) && dt > Real{0})) {
      throw std::invalid_argument{
          "EmPic2D3V::seed_initial_fields: the cylindrical standing mode needs "
          "the positive solver timestep so Bphi can be initialized at t=-dt/2"};
    }
    // Order matters: the derivative below reads the ghost columns, so the
    // configured closure has to have written them first. That is the whole
    // mechanism by which the axis-even / wall-odd parity stops being restated
    // in the deck layer.
    backend::device_synchronize(nullptr);
    fill_field_ghosts();
    backend::device_synchronize(nullptr);

    PicRadialHalfStepSpec half{};
    half.grid = grid_;
    half.fdtd_order = cfg_.fdtd_order;
    half.half_dt = Real{-0.5} * dt;
    half.dr = grid_.dx();
    half.source_component = resolved.component;
    half.target_component = static_cast<int>(PicFieldComponent::bz);
    launch_pic_seed_radial_half_step(half, fields_, status.device_ptr(),
                                     nullptr);
  }

  int host_status = 0;
  status.copy_to_host(&host_status, 1);
  backend::device_synchronize(nullptr);
  if ((host_status & kPicSeedFieldNotFinite) != 0) {
    throw std::overflow_error{
        "EmPic2D3V::seed_initial_fields: the seeded field is not representable "
        "in the solver normalization"};
  }
}

void EmPic2D3V::fill_field_ghosts() {
  for (int side = 0; side < 4; ++side) {
    field_bcs_[side]->fill_ghosts(fields_, static_cast<Side>(side));
  }
}

void EmPic2D3V::correct_field_boundaries_b(Real dt) {
  for (int side = 0; side < 4; ++side) {
    field_bcs_[side]->correct_after_b(fields_, static_cast<Side>(side), dt);
  }
}

void EmPic2D3V::correct_field_boundaries_e(Real dt) {
  for (int side = 0; side < 4; ++side) {
    field_bcs_[side]->correct_after_e(fields_, static_cast<Side>(side), dt);
  }
}

void EmPic2D3V::prime_outflow_corners() {
  if (outflow_corner_mask_ == 0u || outflow_corners_primed_) return;
  ::launch_pic_boundary_outflow_corners(
      grid_, fields_, Real{0}, outflow_corner_mask_,
      outflow_corner_history_.device_ptr(), /*init=*/1,
      is_cylindrical(cfg_.geometry) ? 1 : 0, nullptr);
  outflow_corners_primed_ = true;
}

void EmPic2D3V::correct_outflow_corners(Real dt) {
  if (outflow_corner_mask_ == 0u) return;
  if (!outflow_corners_primed_) {
    throw std::logic_error{
        "EmPic2D3V: outflow corner history was not primed before Ampere"};
  }
  ::launch_pic_boundary_outflow_corners(
      grid_, fields_, dt, outflow_corner_mask_,
      outflow_corner_history_.device_ptr(), /*init=*/0,
      is_cylindrical(cfg_.geometry) ? 1 : 0, nullptr);
  // The diagonal update changed a physical corner sample after the face BCs
  // refreshed their halos. Rebuild all separable side/corner continuations.
  fill_field_ghosts();
}

void EmPic2D3V::apply_particle_bcs_before_deposit(ParticleSpecies& s) {
  // Dispatch through the registry-built IParticleBoundary objects. The former
  // "heisenbug" that made this path fault was a registry factory collision
  // (identical-code folding collapsed the stateless make_unique lambdas, so
  // create() returned the wrong concrete type); it is fixed in
  // core/registry.hpp via type-keyed registration.
  for (int side = 0; side < 4; ++side) {
    // An absorbed particle must remain alive through current deposition so its
    // trajectory carries charge through the boundary face. Killing it here made
    // charge disappear without a boundary current.
    if (cfg_.boundary.particle[side] == "absorbing") continue;
    particle_bcs_[side]->apply(s, static_cast<Side>(side));
  }
}

void EmPic2D3V::apply_absorbing_bcs_after_deposit(ParticleSpecies& s) {
  for (int side = 0; side < 4; ++side) {
    if (cfg_.boundary.particle[side] != "absorbing") continue;
    particle_bcs_[side]->apply(s, static_cast<Side>(side));
  }
}

void EmPic2D3V::prepare_absorbing_bcs_for_deposit(ParticleSpecies& s) {
  const int shape_order = cfg_.shape == "tsc" ? 2 : 1;
  for (int side = 0; side < 4; ++side) {
    if (cfg_.boundary.particle[side] != "absorbing") continue;
    particle_bcs_[side]->prepare_deposit(s, static_cast<Side>(side),
                                         shape_order);
  }
}

void EmPic2D3V::step(Real dt) {
  if (!(std::isfinite(dt) && dt > Real{0})) {
    throw std::invalid_argument{"EmPic2D3V::step: dt must be finite and positive"};
  }
  check_cfl(dt);

  // Public/deck velocities are physical values at t=0, while a seeded magnetic
  // field is already at t=-dt/2.  The first Faraday update must therefore keep
  // its full width dt (B^-1/2 -> B^+1/2), but the first Boris force update spans
  // only 0 -> dt/2.  Its drift then uses the correctly centred v^1/2.
  //
  // After startup, B and velocity are both stored at half steps.  If the next
  // position-step width changes (notably for advance()'s exact final step), both
  // updates span from t^n-dt_prev/2 to t^n+dt/2, hence
  // (dt_prev+dt)/2.  The old and new half-step B values bracket t^n
  // asymmetrically and are interpolated with the opposite interval widths.
  Real magnetic_dt = dt;
  Real force_dt = Real{0.5} * dt;
  const bool has_charged_particles = std::any_of(
      species_.cbegin(), species_.cend(),
      [](const ParticleSpecies& species) {
        return species.size() != 0 && species.charge() != Real{0};
      });
  if (has_charged_particles &&
      !(std::isfinite(force_dt) && force_dt > Real{0})) {
    throw std::overflow_error{
        "EmPic2D3V::step: first charged-particle half timestep is not "
        "representable"};
  }
  Real previous_b_weight = Real{0.5};
  Real current_b_weight = Real{0.5};
  if (has_previous_dt_) {
    magnetic_dt = std::midpoint(previous_dt_, dt);
    force_dt = magnetic_dt;
    if (!(std::isfinite(magnetic_dt) && magnetic_dt > Real{0})) {
      throw std::overflow_error{
          "EmPic2D3V::step: centered leapfrog timestep is not representable"};
    }
    if (previous_dt_ >= dt) {
      const Real ratio = dt / previous_dt_;
      current_b_weight = Real{1} / (Real{1} + ratio);
      previous_b_weight = ratio * current_b_weight;
    } else {
      const Real ratio = previous_dt_ / dt;
      previous_b_weight = Real{1} / (Real{1} + ratio);
      current_b_weight = ratio * previous_b_weight;
    }
    check_cfl(magnetic_dt);
  }
  // Validate/materialize the initial charge before locking species mutation.
  // A rejected non-neutral periodic setup has not changed leapfrog state and
  // may still be repaired by adding its missing counter-species.
  ensure_charge_density();
  // From here onward B/velocity acquire leapfrog half-step staggering. Species
  // insertion or host replacement after this point would require an explicit
  // restaggering operation, which this API intentionally does not imply.
  evolution_started_ = true;
  // Clear the current accumulators asynchronously on the default stream. The
  // deposit and every downstream kernel also run on the default stream, so the
  // clear is correctly ordered before them without a host-side block (the old
  // synchronous hipMemset stalled the host every step).
  backend::device_memset_async(current_.jx.device_ptr(), 0, current_.jx.bytes(), nullptr);
  backend::device_memset_async(current_.jy.device_ptr(), 0, current_.jy.bytes(), nullptr);
  backend::device_memset_async(current_.jz.device_ptr(), 0, current_.jz.bytes(), nullptr);

  // The FDTD stencil reads neighbours through ghost cells, so every per-side
  // ghost fill must run before each curl. Periodic sides copy the opposite
  // interior edge, PEC sides apply their parity continuation, and outflow sides
  // apply their linear continuation.
  fill_field_ghosts();
  ::launch_pic_copy_b(grid_, fields_, previous_b_, nullptr);
  field_solver_->advance_b(fields_, magnetic_dt);
  // Refresh PEC/outflow boundary continuations after the B update; periodic
  // correction remains a no-op because its pre-curl wrap is already complete.
  correct_field_boundaries_b(magnetic_dt);
  // Capture characteristic wall/corner history while E is still at t^n. The
  // first post-Ampere correction must compare against this state, not merely
  // seed itself from E^{n+1}.
  prime_outflow_corners();
  for (auto& s : species_) {
    pusher_->push(s, fields_, external_fields_, previous_b_, force_dt, dt,
                  previous_b_weight, current_b_weight);
    apply_particle_bcs_before_deposit(s);
    prepare_absorbing_bcs_for_deposit(s);
    deposit_->deposit(s, current_, dt);
    apply_absorbing_bcs_after_deposit(s);
  }
  deposit_charge_density(next_charge_);
  // Coordinate/range failures cause a deposit kernel to omit that particle's
  // contribution. Drain the sticky flag before Ampere can consume an incomplete
  // current, so both direct step() callers and Python runs fail deterministically
  // at the offending step rather than evolving corrupted fields until a cadence.
  check_deposit_overflow();
  // On reflecting (specular) sides the deposit left boundary-crossing current in
  // the ghost cells; the BC's fold_current hook folds it back into the interior
  // as image current before the filter / E-update read J. Periodic/absorbing
  // BCs leave fold_current a no-op.
  for (int side = 0; side < 4; ++side) {
    particle_bcs_[side]->fold_current(current_, static_cast<Side>(side));
  }
  filters_.apply(current_, cfg_.boundary, is_cylindrical(cfg_.geometry));
  if (cfg_.fdtd_order == 4) {
    if (is_cylindrical(cfg_.geometry)) {
      ::launch_pic_current_correct_cyl_order4(
          grid_, current_, current_rhs_x_.device_ptr(), current_rhs_y_.device_ptr(),
          current_iter_x_.device_ptr(), current_iter_y_.device_ptr(),
          current_ghost_mode(*field_bcs_[0], cfg_.boundary.field[0]),
          current_ghost_mode(*field_bcs_[1], cfg_.boundary.field[1]),
          current_ghost_mode(*field_bcs_[2], cfg_.boundary.field[2]),
          current_ghost_mode(*field_bcs_[3], cfg_.boundary.field[3]), nullptr);
    } else {
      ::launch_pic_current_correct_order4(
          grid_, current_, current_rhs_x_.device_ptr(), current_rhs_y_.device_ptr(),
          current_iter_x_.device_ptr(), current_iter_y_.device_ptr(),
          current_ghost_mode(*field_bcs_[0], cfg_.boundary.field[0]),
          current_ghost_mode(*field_bcs_[1], cfg_.boundary.field[1]),
          current_ghost_mode(*field_bcs_[2], cfg_.boundary.field[2]),
          current_ghost_mode(*field_bcs_[3], cfg_.boundary.field[3]), nullptr);
    }
  }
  // Deposits wrap periodic normal flux into the unique low face. Filters and
  // the order-four compact inverse may then touch the current arrays, so restore
  // the duplicate physical high face only here, immediately before Ampere reads
  // it. This applies to order two as well as order four.
  ::launch_pic_current_periodic_high_faces(
      grid_, current_, periodic_x_ ? 1 : 0, periodic_y_ ? 1 : 0, nullptr);
  fill_field_ghosts();
  // The deposit flag above covers only deposition itself. Wall foldback,
  // filters, the compact order-four solve, periodic-face restoration, and
  // charge background/foldback all perform additional floating-point
  // arithmetic. Reject a non-finite source before Ampere consumes current or
  // next_charge_ becomes the live diagnostic field.
  ::launch_pic_validate_finite_sources(
      grid_, &current_, &next_charge_, source_finite_error_.device_ptr(),
      nullptr);
  field_solver_->advance_e(fields_, current_, dt);
  // Outflow Mur reads the just-updated adjacent interior E node, so the E-side
  // correction runs after advance_e. PEC refreshes its parity; periodic is a no-op.
  correct_field_boundaries_e(dt);
  correct_outflow_corners(dt);
  std::swap(charge_.values, next_charge_.values);
  previous_dt_ = dt;
  has_previous_dt_ = true;

  // Reclaim slots vacated by absorbing boundaries on a fixed cadence so the
  // push/deposit/gather kernels stop iterating over dead particles. Only
  // meaningful when a boundary can actually kill particles; periodic/specular
  // runs never shrink, so skip the work entirely.
  ++step_count_;
  constexpr std::size_t kCompactEvery = 64;
  const bool cadence_hit = step_count_ % kCompactEvery == 0;
  if (has_absorbing_boundary() && cadence_hit) {
    for (auto& s : species_) {
      ::launch_pic_particle_compact(s, nullptr);
    }
  }
}

void EmPic2D3V::check_deposit_overflow() {
  for (auto& s : species_) {
    ::launch_pic_deposit_overflow_check(s, nullptr);
  }
}

void EmPic2D3V::finalize() {
  check_deposit_overflow();
}

void EmPic2D3V::advance(Real t_end, Real dt) {
  if (!(std::isfinite(dt) && dt > Real{0})) {
    throw std::invalid_argument{"EmPic2D3V::advance: dt must be finite and positive"};
  }
  check_cfl(dt);
  if (!(std::isfinite(t_end) && t_end >= Real{0})) {
    throw std::invalid_argument{
        "EmPic2D3V::advance: t_end must be finite and non-negative"};
  }
  if (t_end > Real{0} && !std::isfinite(t_end / dt)) {
    throw std::overflow_error{
        "EmPic2D3V::advance: t_end/dt is not representable"};
  }

  // Advance by the remaining interval instead of converting floor(t_end/dt)
  // to an integer.  The quotient may exceed LONG_MAX, and a quotient rounded
  // upward can otherwise schedule a full step past t_end.  Assign the exact
  // endpoint after the clipped final step so accumulated roundoff cannot create
  // a spurious extra iteration.
  Real elapsed = Real{0};
  while (elapsed < t_end) {
    const Real remaining = t_end - elapsed;
    const Real dt_step = std::min(dt, remaining);
    if (!(std::isfinite(dt_step) && dt_step > Real{0})) {
      throw std::runtime_error{
          "EmPic2D3V::advance: timestep made no forward progress"};
    }
    const Real next = elapsed + dt_step;
    if (!(next > elapsed)) {
      throw std::runtime_error{
          "EmPic2D3V::advance: floating-point time made no forward progress"};
    }
    step(dt_step);
    elapsed = dt_step == remaining ? t_end : next;
  }
  finalize();
}

Real EmPic2D3V::cfl_limit() const {
  // The volume-weighted cylindrical radial derivative has the same operator
  // norm as its Cartesian staggered counterpart at each supported order; the
  // regular axis row does not tighten the bound.
  if (is_cylindrical(cfg_.geometry)) {
    return cyl_cfl_dt(grid_, cfg_.fdtd_order, Real{1});
  }
  return cfl_dt(grid_, cfg_.fdtd_order, Real{1});
}

void EmPic2D3V::check_cfl(Real dt) const {
  // A dt above the Yee CFL limit makes the FDTD update diverge exponentially, so
  // reject it rather than stepping into an unstable run.
  if (!(std::isfinite(dt) && dt > Real{0})) {
    throw std::invalid_argument{"EmPic2D3V: dt must be finite and positive"};
  }
  if (dt > cfl_limit()) {
    throw std::invalid_argument{
        "EmPic2D3V: dt exceeds the CFL stability limit for this grid and scheme"};
  }
}

}  // namespace quasar::pic
