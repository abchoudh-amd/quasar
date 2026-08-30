#include "quasar/physics/mhd/mhd_solver.hpp"

#include "quasar/numerics/mhd_background_metrics.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/core/registry.hpp"
// The solver owns no ICtScheme / IRiemannSolver member, but it still validates
// that the configured names resolve in those registries, so both interfaces are
// needed here rather than in the public header.
#include "quasar/numerics/ct_scheme.hpp"
#include "quasar/numerics/mhd_background_profile.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/numerics/riemann_solver.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quasar::mhd {

namespace {


// Halo -> spatial order now lives beside IFluxReconstruction::required_nghost()
// in numerics/flux_reconstruction.hpp, so the Python deck layer can read the
// same mapping through a binding instead of mirroring it.
using numerics::reconstruction_order_from_nghost;

template <class Base>
void require_registered_name(const std::string& name, const char* what) {
  if (name.empty() || !Registry<Base>::instance().contains(name)) {
    throw std::invalid_argument{
        std::string{"MhdSolver2D: unknown "} + what + " '" + name + "'"};
  }
}

template <class Base>
int registered_boundary_mode(const std::string& name, const char* what) {
  require_registered_name<Base>(name, what);
  return Registry<Base>::instance().create(name)->ghost_continuation_mode();
}

// Validate every scalar/configuration value before registry construction or any
// grid-sized device allocation. Python performs the same checks for a deck, but
// MhdSolver2D is also a public C++ API and must not send invalid thermodynamics
// or CFL values into device kernels.
MhdConfig validate_config(MhdConfig cfg) {
  cfg.grid.validate();
  if (!(std::isfinite(cfg.gamma) && cfg.gamma > Real{1})) {
    throw std::invalid_argument{
        "MhdSolver2D: gamma must be finite and greater than one"};
  }
  if (!(std::isfinite(cfg.rho_floor) && cfg.rho_floor > Real{0})) {
    throw std::invalid_argument{
        "MhdSolver2D: rho_floor must be finite and positive"};
  }
  if (!(std::isfinite(cfg.p_floor) && cfg.p_floor > Real{0})) {
    throw std::invalid_argument{
        "MhdSolver2D: p_floor must be finite and positive"};
  }
  if (!(std::isfinite(cfg.cfl) && cfg.cfl > Real{0} && cfg.cfl <= Real{1})) {
    throw std::invalid_argument{
        "MhdSolver2D: cfl must be finite and in (0,1]"};
  }
  if (cfg.geometry != "cartesian" && cfg.geometry != "cylindrical") {
    throw std::invalid_argument{
        "MhdSolver2D: geometry must be 'cartesian' or 'cylindrical'"};
  }
  require_registered_name<numerics::IFluxReconstruction>(
      cfg.reconstruction, "reconstruction scheme");
  require_registered_name<numerics::IRiemannSolver>(
      cfg.riemann, "Riemann solver");
  require_registered_name<numerics::ISsprkIntegrator>(
      cfg.integrator, "integrator");
  require_registered_name<numerics::ICtScheme>(cfg.ct, "CT scheme");
  require_registered_name<numerics::IPositivityLimiter>(
      cfg.positivity, "positivity limiter");
  // Cylindrical MP5/MP7 use the radius-dependent finite-volume coefficients
  // built by numerics/radial_moments.hpp; axial stencils remain Cartesian.
  // The device residual currently has one concrete algorithm per numerics axis.
  // A registry entry alone must not be accepted and then silently ignored.
  if (cfg.riemann != "hlld") {
    throw std::invalid_argument{
        "MhdSolver2D: device evolution supports only riemann='hlld'"};
  }
  if (cfg.ct != "fd_ct_christlieb") {
    throw std::invalid_argument{
        "MhdSolver2D: device evolution supports only ct='fd_ct_christlieb'"};
  }
  // compute_residual() launches a built-in MUSCL/MP5/MP7 device kernel selected
  // by reconstruction_order(), which maps the scheme's required_nghost() back to
  // order 2/5/7. A custom IFluxReconstruction's reconstruct_faces() is therefore
  // never called on the evolution path: accepting one would run built-in MP5
  // under the custom scheme's name. Reject by name rather than silently
  // substituting a different algorithm.
  if (cfg.reconstruction != "muscl_minmod" && cfg.reconstruction != "mp5" &&
      cfg.reconstruction != "mp7") {
    throw std::invalid_argument{
        "MhdSolver2D: device evolution supports only "
        "reconstruction='muscl_minmod', 'mp5', or 'mp7'; a custom registered "
        "reconstruction would be silently replaced by the built-in kernel of "
        "the same halo width"};
  }
  // The integrator axis is a SEQUENCING seam, not a coefficient-carrying one:
  // ISsprkIntegrator::advance() is genuinely invoked, but combine_stage() owns
  // the Shu-Osher weights and accepts only stages 0/1/2 (see the structural
  // n_stages() check in the constructor, which needs the built object).
  std::array<int, 4> fluid_modes{};
  std::array<int, 4> field_modes{};
  for (int side = 0; side < 4; ++side) {
    fluid_modes[side] = registered_boundary_mode<boundary::IMhdFluidBoundary>(
        cfg.boundary.fluid[side], "fluid boundary");
    field_modes[side] = registered_boundary_mode<boundary::IMhdFieldBoundary>(
        cfg.boundary.field[side], "field boundary");
    const bool fluid_periodic = fluid_modes[side] == 0;
    const bool field_periodic = field_modes[side] == 0;
    const bool fluid_internal = fluid_modes[side] == 4;
    const bool field_internal = field_modes[side] == 4;
    if (fluid_periodic != field_periodic || fluid_internal != field_internal) {
      throw std::invalid_argument{
          "MhdSolver2D: fluid and field topology must match on every side"};
    }
  }
  for (int axis = 0; axis < 2; ++axis) {
    const int lo = 2 * axis;
    const int hi = lo + 1;
    const bool fluid_lo = fluid_modes[lo] == 0;
    const bool fluid_hi = fluid_modes[hi] == 0;
    const bool field_lo = field_modes[lo] == 0;
    const bool field_hi = field_modes[hi] == 0;
    if (fluid_lo != fluid_hi || field_lo != field_hi) {
      throw std::invalid_argument{
          "MhdSolver2D: periodic boundaries must be selected on both sides of an axis"};
    }
  }
  const bool cylindrical = cfg.geometry == "cylindrical";
  if (cylindrical && cfg.grid.origin_x < Real{0}) {
    throw std::invalid_argument{
        "MhdSolver2D: cylindrical geometry requires a non-negative radial origin"};
  }
  for (int side = 0; side < 4; ++side) {
    const bool fluid_axis = fluid_modes[side] == 3;
    const bool field_axis = field_modes[side] == 3;
    if (fluid_axis != field_axis) {
      throw std::invalid_argument{
          "MhdSolver2D: fluid and field axis closures must be selected together"};
    }
    if ((fluid_axis || field_axis) &&
        (!cylindrical || cfg.grid.origin_x != Real{0} || side != 0)) {
      throw std::invalid_argument{
          "MhdSolver2D: the 'axis' boundary is valid only at cylindrical r=0 (x_lo)"};
    }
  }
  if (cylindrical && cfg.grid.origin_x == Real{0} &&
      (fluid_modes[0] != 3 || field_modes[0] != 3)) {
    throw std::invalid_argument{
        "MhdSolver2D: cylindrical r=0 requires paired fluid/field 'axis' boundaries"};
  }
  if (cylindrical &&
      (fluid_modes[0] == 0 || fluid_modes[1] == 0)) {
    throw std::invalid_argument{
        "MhdSolver2D: the radial axis cannot use periodic boundaries"};
  }
  if (cfg.background.enabled) {
    if (!(std::isfinite(cfg.background.bx0) &&
          std::isfinite(cfg.background.by0) &&
          std::isfinite(cfg.background.bz0) &&
          std::isfinite(cfg.background.profile_scale))) {
      throw std::invalid_argument{
          "MhdSolver2D: background parameters must be finite"};
    }
    if (cfg.background.profile.empty() ||
        !Registry<numerics::IMhdBackgroundProfile>::instance().contains(
            cfg.background.profile)) {
      throw std::invalid_argument{
          "MhdSolver2D: unknown MHD background profile '" +
          cfg.background.profile + "'"};
    }
    if (cfg.background.profile != "uniform" &&
        (cfg.background.bx0 != Real{0} || cfg.background.by0 != Real{0} ||
         cfg.background.bz0 != Real{0})) {
      throw std::invalid_argument{
          "MhdSolver2D: bx0/by0/bz0 are valid only for background profile "
          "'uniform'"};
    }
    for (const auto& [name, value] : cfg.background.params) {
      if (name.empty()) {
        throw std::invalid_argument{
            "MhdSolver2D: background parameter names must be non-empty"};
      }
      if (!std::isfinite(value)) {
        throw std::invalid_argument{
            "MhdSolver2D: background parameter '" + name +
            "' must be finite"};
      }
    }
  }
  return cfg;
}

// Resolve the working-grid ghost halo authoritatively from the reconstruction
// scheme. A deck may leave grid.nghost at 0 (let the solver size it); if it set
// a positive value it must be at least the scheme's required halo, else the
// reconstruction would read past the allocated ghosts.
Grid2D resolve_working_grid(const Grid2D& deck_grid, int required,
                            const std::string& geometry,
                            bool internal_radial_low) {
  Grid2D g = deck_grid;
  if (deck_grid.nghost > 0 && deck_grid.nghost < required) {
    throw std::invalid_argument{
        "MhdSolver2D: grid nghost (" + std::to_string(deck_grid.nghost) +
        ") is smaller than the reconstruction scheme's required halo (" +
        std::to_string(required) + ")"};
  }
  g.nghost = std::max(deck_grid.nghost, required);
  g.validate();
  // An annular grid has no r=0 parity closure. Every radial coordinate reached
  // by reconstruction/background sampling must therefore remain strictly
  // positive, including the low ghost halo. Crossing r=0 would reinterpret
  // negative radii as ordinary Cartesian coordinates in annular formulas.
  if (geometry == "cylindrical" && g.origin_x > Real{0}) {
    const Real padded_r_lo =
        g.origin_x - static_cast<Real>(g.nghost) * g.dx();
    const bool valid = std::isfinite(padded_r_lo) &&
        (internal_radial_low ? padded_r_lo >= Real{0}
                             : padded_r_lo > Real{0});
    if (!valid) {
      throw std::invalid_argument{
          internal_radial_low
              ? "MhdSolver2D: an internal annular tile requires "
                "origin_x - nghost*dr >= 0 so its reconstruction halo does "
                "not cross the global axis"
              : "MhdSolver2D: annular geometry requires "
                "origin_x - nghost*dr > 0 so the full reconstruction halo "
                "stays at positive radius"};
    }
  }
  return g;
}


// Boundary mode codes handed to the device sweep. The kernel ABI takes ints so
// the backend never sees the boundary axis's string vocabulary; modes it does
// not constrain map to 0 and are skipped.
int background_boundary_mode(const std::string& mode) {
  if (mode == "periodic") return 1;
  if (mode == "wall") return 2;
  if (mode == "axis") return 3;
  return 0;
}

// Report the closure rule the sequential sweep would have reported first. The
// device sweep returns every violated rule at once because it visits the sides
// in parallel; picking the lowest set bit in side order restores a stable,
// reproducible message rather than whichever thread happened to finish first.
void throw_on_boundary_violation(unsigned violated) {
  struct RuleMessage {
    unsigned bit;
    const char* rule;
  };
  static constexpr RuleMessage kRules[] = {
      {mhd_background_rule_periodic_x, "the periodic x boundary"},
      {mhd_background_rule_axis_parity, "the cylindrical axis parity"},
      {mhd_background_rule_axis_constraint, "the cylindrical axis constraint"},
      {mhd_background_rule_x_wall_parity, "the x-wall parity"},
      {mhd_background_rule_x_wall_normal, "the x-wall normal constraint"},
      {mhd_background_rule_periodic_y, "the periodic y boundary"},
      {mhd_background_rule_y_wall_parity, "the y-wall parity"},
      {mhd_background_rule_y_wall_normal, "the y-wall normal constraint"},
  };
  for (const RuleMessage& entry : kRules) {
    if ((violated & entry.bit) != 0u) {
      throw std::invalid_argument{
          std::string{"MhdSolver2D: background field is incompatible with "} +
          entry.rule};
    }
  }
}

// Discrete solenoidality of a prescribed background, plus the closure-parity
// rules its padded samples must already satisfy. The sweeps run on device; what
// happens here is the comparison against the tolerance and the throw.
void validate_background_field(
    const MhdBackgroundField<Real>& b0, const Grid2D& grid, bool cylindrical,
    const boundary::MhdBoundarySpec& boundary_spec) {
  const MhdBackgroundSolenoidalResult solenoidal =
      launch_mhd_validate_background_solenoidal(
          b0, grid, cylindrical ? 1 : 0, nullptr);
  if (solenoidal.nonfinite_sample != 0) {
    throw std::invalid_argument{
        "MhdSolver2D: background field must contain only finite values"};
  }
  const Real relative_linf = numerics::normalized_scaled_ratio(
      solenoidal.residual_linf, solenoidal.directional_scale_linf);
  if (!(std::isfinite(relative_linf) &&
        relative_linf <= numerics::kDiscreteSolenoidalTolerance)) {
    throw std::invalid_argument{
        "MhdSolver2D: background field is not discretely divergence-free"};
  }

  const int modes[4] = {
      background_boundary_mode(boundary_spec.field[0]),
      background_boundary_mode(boundary_spec.field[1]),
      background_boundary_mode(boundary_spec.field[2]),
      background_boundary_mode(boundary_spec.field[3]),
  };
  throw_on_boundary_violation(
      launch_mhd_validate_background_boundaries(b0, grid, modes, nullptr)
          .violated_rules);
}

// Defense in depth against a curl-free assertion that plainly contradicts its
// own samples. This is never a way to INFER the proof: the authorization to
// omit B0's static stress must come from a trusted analytic or projection
// construction, and omitting it cell-by-cell would destroy shared-face
// conservation at the boundary of the selected patch.
void validate_background_curl_free(
    const MhdBackgroundField<Real>& b0, const Grid2D& grid, bool cylindrical) {
  const MhdBackgroundCurlFreeResult curl =
      launch_mhd_validate_background_curl_free(
          b0, grid, cylindrical ? 1 : 0, nullptr);
  if (curl.negative_interior_radius != 0) {
    throw std::invalid_argument{
        "MhdSolver2D: curl_free cylindrical background encountered "
        "a negative interior radius"};
  }
  if (curl.single_derivative_is_zero == 0 ||
      !(std::isfinite(curl.relative_linf) &&
        curl.relative_linf <= numerics::kDiscreteCurlFreeTolerance)) {
    throw std::invalid_argument{
        "MhdSolver2D: background curl_free assertion failed the discrete "
        "curl check"};
  }
}

// Resolve a pluggable scheme by registry name. Throws std::invalid_argument on an
// unknown deck name (Registry::create throws std::out_of_range, which we rephrase
// for the deck). Used for every MHD scheme axis, including the reconstruction
// scheme that is built first because its required_nghost() fixes the working grid
// (so MhdSolver2D's member initializer list can size every field/register at the
// resolved grid).
template <class Base>
std::unique_ptr<Base> make_scheme(const std::string& name, const char* what) {
  try {
    auto p = Registry<Base>::instance().create(name);
    if (!p) {
      throw std::invalid_argument{std::string{"MhdSolver2D: unknown "} + what + " '" + name + "'"};
    }
    return p;
  } catch (const std::out_of_range&) {
    throw std::invalid_argument{std::string{"MhdSolver2D: unknown "} + what + " '" + name + "'"};
  }
}

}  // namespace

MhdSolver2D::MhdSolver2D(MhdConfig cfg)
  : cfg_{validate_config(std::move(cfg))},
    reconstruction_{make_scheme<numerics::IFluxReconstruction>(cfg_.reconstruction,
                                                              "reconstruction scheme")},
    grid_{resolve_working_grid(
        cfg_.grid, reconstruction_->required_nghost(), cfg_.geometry,
        registered_boundary_mode<boundary::IMhdFluidBoundary>(
            cfg_.boundary.fluid[0], "fluid boundary") == 4 &&
            registered_boundary_mode<boundary::IMhdFieldBoundary>(
                cfg_.boundary.field[0], "field boundary") == 4)},
    radial_tables_{cfg_.geometry == "cylindrical"
                       ? numerics::RadialTables{
                             grid_, reconstruction_order_from_nghost(
                                        reconstruction_->required_nghost())}
                       : numerics::RadialTables{}},
    rk_{MhdField2D<Real>{grid_}, MhdField2D<Real>{grid_}, MhdField2D<Real>{grid_}},
    step_backup_{grid_},
    request_backup_{grid_},
    residual_{grid_},
    flux_x_{grid_},
    flux_y_{grid_},
    momentum_flux_x_{grid_},
    momentum_flux_y_{grid_},
    ifx_{grid_, 0},
    ify_{grid_, 1},
    emf_{grid_} {
  // Field-split background B0: allocate at the working grid and mark active iff
  // the deck enabled it. Otherwise leave b0_ default-constructed (inactive =>
  // zero-B0 fast path that is bit-identical to the no-background solver).
  // Enabled native C++ configurations are complete in their own right: resolve
  // and sample the named profile here instead of relying on the Python CLI to
  // overwrite initially-zero buffers after construction.
  if (cfg_.background.enabled) {
    b0_ = MhdBackgroundField<Real>{grid_};
    b0_.active = true;

    auto profile = make_scheme<numerics::IMhdBackgroundProfile>(
        cfg_.background.profile, "MHD background profile");
    for (const auto& [name, value] : cfg_.background.params) {
      if (!profile->set_parameter(name, value)) {
        throw std::invalid_argument{
            "MhdSolver2D: unknown parameter '" + name +
            "' for MHD background profile '" + cfg_.background.profile + "'"};
      }
    }
    if (cfg_.background.profile == "uniform") {
      const bool bx_ok = profile->set_parameter("bx0", cfg_.background.bx0);
      const bool by_ok = profile->set_parameter("by0", cfg_.background.by0);
      const bool bz_ok = profile->set_parameter("bz0", cfg_.background.bz0);
      if (!(bx_ok && by_ok && bz_ok)) {
        throw std::logic_error{
            "MhdSolver2D: registered uniform background profile does not "
            "accept bx0/by0/bz0"};
      }
    }

    const bool cylindrical = is_cylindrical();
    const std::size_t n = grid_.storage_size();
    std::vector<Real> b0x(n), b0y(n), b0z(n);
    for (int j = -grid_.nghost; j < grid_.ny + grid_.nghost; ++j) {
      for (int i = -grid_.nghost; i < grid_.nx + grid_.nghost; ++i) {
        const Real xc = cylindrical
            ? grid_.r_at_cell_center(i)
            : grid_.x_at_cell_center(i);
        const Real yc = grid_.y_at_cell_center(j);
        const Real xf = cylindrical
            ? grid_.r_at_edge(i)
            : grid_.origin_x + static_cast<Real>(i) * grid_.dx();
        const Real yf = grid_.origin_y + static_cast<Real>(j) * grid_.dy();
        if (!(std::isfinite(xc) && std::isfinite(yc) &&
              std::isfinite(xf) && std::isfinite(yf))) {
          throw std::overflow_error{
              "MhdSolver2D: padded background coordinates are not representable"};
        }
        const std::size_t k = grid_.index(i, j);
        b0x[k] = cfg_.background.profile_scale * profile->sample(0, xf, yc);
        b0y[k] = cfg_.background.profile_scale * profile->sample(1, xc, yf);
        b0z[k] = cfg_.background.profile_scale * profile->sample(2, xc, yc);
        if (!(std::isfinite(b0x[k]) && std::isfinite(b0y[k]) &&
              std::isfinite(b0z[k]))) {
          throw std::invalid_argument{
              "MhdSolver2D: background profile produced a non-finite sample"};
        }
      }
    }

    // Upload first, then validate the device buffers. The checks used to run on
    // these host vectors before the seed; running them after means the sweep
    // reads exactly the bytes the solver will use, so a defect introduced by
    // the upload path itself cannot slip past.
    seed_background("b0x", b0x);
    seed_background("b0y", b0y);
    seed_background("b0z", b0z);

    validate_background_field(b0_, grid_, cylindrical, cfg_.boundary);
    const bool profile_curl_free =
        !cylindrical && profile->globally_curl_free();
    const bool globally_curl_free =
        cfg_.background.curl_free || profile_curl_free;
    if (globally_curl_free) {
      validate_background_curl_free(b0_, grid_, cylindrical);
    }

    b0_.globally_curl_free = globally_curl_free;
    background_validated_ = true;
  }

  // Resolve the remaining schemes by registry string (no if/else over types).
  // The Riemann and CT axes are deliberately absent here: the constructor gate
  // above already pinned each to its single supported name, and the residual
  // path calls launch_mhd_hlld_flux / launch_mhd_ct_emf_* directly. Building an
  // IRiemannSolver or ICtScheme that nothing dereferences would advertise a
  // dispatch seam the device path does not use. Their registered classes stay
  // in the registry as the host-side harnesses those kernels are tested through
  // (tests/unit/numerics/test_hlld_riemann.cpp, test_ct_divergence.cpp), and
  // require_registered_name above still proves the registrations link.
  integrator_ = make_scheme<numerics::ISsprkIntegrator>(cfg_.integrator, "integrator");
  // combine_stage() owns the Shu-Osher weights and implements exactly the
  // three-stage SSP-RK3 tableau, rejecting any stage index outside {0,1,2}. An
  // integrator advertising a different stage count would therefore either throw
  // partway through a step or silently run with SSPRK3's coefficients. Reject it
  // here instead, while the failure is still a construction-time error.
  if (integrator_->n_stages() != 3) {
    throw std::invalid_argument{
        "MhdSolver2D: integrator '" + cfg_.integrator + "' declares " +
        std::to_string(integrator_->n_stages()) +
        " stages, but combine_stage applies the fixed three-stage SSPRK3 "
        "Shu-Osher coefficients; only a 3-stage integrator is supported"};
  }
  positivity_ = make_scheme<numerics::IPositivityLimiter>(cfg_.positivity, "positivity limiter");

  // Per-side fluid + field boundaries, built through the registry (the documented
  // pluggable path); concrete BCs self-register in src/physics/mhd/mhd_boundary.cpp.
  for (int side = 0; side < 4; ++side) {
    fluid_bcs_[side] =
        make_scheme<boundary::IMhdFluidBoundary>(cfg_.boundary.fluid[side], "fluid boundary");
    field_bcs_[side] =
        make_scheme<boundary::IMhdFieldBoundary>(cfg_.boundary.field[side], "field boundary");
  }
}

backend::DeviceBuffer<Real>& MhdSolver2D::component_buffer(MhdField2D<Real>& f,
                                                          std::string_view component) {
  if (component == "rho") return f.rho;
  if (component == "mx") return f.mx;
  if (component == "my") return f.my;
  if (component == "mz") return f.mz;
  if (component == "energy") return f.energy;
  if (component == "bx" || component == "bx_face") return f.bx_face;
  if (component == "by" || component == "by_face") return f.by_face;
  if (component == "bz" || component == "bz_cell") return f.bz_cell;
  throw std::invalid_argument{"MhdSolver2D: unknown state component '" +
                              std::string{component} + "'"};
}

void MhdSolver2D::seed_state(std::string_view component, const std::vector<Real>& host_buf) {
  auto& buf = component_buffer(rk_[0], component);
  if (host_buf.size() != buf.size()) {
    throw std::invalid_argument{
        "MhdSolver2D::seed_state: host buffer size (" + std::to_string(host_buf.size()) +
        ") does not match the component storage size (" + std::to_string(buf.size()) + ")"};
  }
  if (!std::all_of(host_buf.begin(), host_buf.end(),
                   [](Real value) { return std::isfinite(value); })) {
    throw std::invalid_argument{
        "MhdSolver2D::seed_state: host buffer must contain only finite values"};
  }
  buf.copy_from_host(host_buf.data(), host_buf.size());
  // Direct register write with no ghost refill, so it does not pass through the
  // fill_ghosts invalidation hook. Invalidate explicitly.
  invalidate_interface_cache();
  // Component-wise seeding is external provenance. The complete state is
  // strictly revalidated before use; a later successful internal step may earn
  // solver-owned provenance again only if no retainable mutable view exists.
  live_state_solver_owned_ = false;
}

void MhdSolver2D::seed_background(std::string_view component,
                                  const std::vector<Real>& host_buf) {
  if (!b0_.active) {
    throw std::logic_error{
        "MhdSolver2D::seed_background: the field-split background is not enabled "
        "(set background.enabled in the deck before seeding B0)"};
  }
  // Map the deck-facing spelling to its background buffer, mirroring the
  // magnetic-component aliases accepted by seed_state.
  backend::DeviceBuffer<Real>* buf = nullptr;
  if (component == "b0x" || component == "b0x_face") {
    buf = &b0_.b0x_face;
  } else if (component == "b0y" || component == "b0y_face") {
    buf = &b0_.b0y_face;
  } else if (component == "b0z" || component == "b0z_cell") {
    buf = &b0_.b0z_cell;
  } else {
    throw std::invalid_argument{"MhdSolver2D::seed_background: unknown background "
                                "component '" + std::string{component} + "'"};
  }
  if (host_buf.size() != buf->size()) {
    throw std::invalid_argument{
        "MhdSolver2D::seed_background: host buffer size (" +
        std::to_string(host_buf.size()) +
        ") does not match the component storage size (" +
        std::to_string(buf->size()) + ")"};
  }
  if (!std::all_of(host_buf.begin(), host_buf.end(),
                   [](Real value) { return std::isfinite(value); })) {
    throw std::invalid_argument{
        "MhdSolver2D::seed_background: host buffer must contain only finite values"};
  }
  buf->copy_from_host(host_buf.data(), host_buf.size());
  // Reconstruction consumes B0, so a background edit invalidates the cached
  // interface states even though no state register moved.
  invalidate_interface_cache();
  // Component-wise overrides no longer carry an analytic profile's proof. An
  // explicit caller may retain only the config-level curl_free assertion; the
  // completed three-component field is independently checked before use.
  b0_.globally_curl_free = cfg_.background.curl_free;
  background_validated_ = false;
}

bool MhdSolver2D::has_background() const noexcept { return cfg_.background.enabled; }

void MhdSolver2D::ensure_background_solenoidal() const {
  if (!b0_.active || background_validated_) return;

  // No download: the sweeps read the resident background directly.
  validate_background_field(b0_, grid_, is_cylindrical(), cfg_.boundary);
  if (b0_.globally_curl_free) {
    validate_background_curl_free(b0_, grid_, is_cylindrical());
  }
  background_validated_ = true;
}

std::vector<Real> MhdSolver2D::state_component_to_host(std::string_view component) const {
  // const-correct read: component_buffer returns a mutable reference even though
  // this path only stages bytes from the live field.
  auto& self = const_cast<MhdSolver2D&>(*this);
  const auto& buf = component_buffer(self.rk_[0], component);
  std::vector<Real> out(buf.size());
  buf.copy_to_host(out.data(), out.size());
  if (component == "bx" || component == "by") {
    const std::vector<Real> face = out;
    const numerics::RadialTablesView radial_tables =
        radial_tables_.host_view();
    const int ilo = -grid_.nghost;
    const int ihi = grid_.nx + grid_.nghost;
    const int jlo = -grid_.nghost;
    const int jhi = grid_.ny + grid_.nghost;
    if (component == "bx") {
      for (int j = jlo; j < jhi; ++j) {
        for (int i = ilo; i < ihi; ++i) {
          out[grid_.index(i, j)] =
              cell_bx(grid_, face.data(), i, j, radial_tables);
        }
      }
    } else {
      for (int j = jlo; j < jhi; ++j) {
        for (int i = ilo; i < ihi; ++i) {
          out[grid_.index(i, j)] =
              cell_by(grid_, face.data(), i, j, radial_tables);
        }
      }
    }
  }
  return out;
}

int MhdSolver2D::reconstruction_order() const {
  // Map the scheme's required halo to a spatial order for the device path:
  //   nghost 2 -> order 2 (muscl_minmod), 3 -> 5 (mp5), 4 -> 7 (mp7).
  return reconstruction_order_from_nghost(reconstruction_->required_nghost());
}

void MhdSolver2D::fill_ghosts(MhdField2D<Real>& u) const {
  for (int side = 0; side < 4; ++side) {
    fluid_bcs_[side]->fill_ghosts(u, static_cast<Side>(side));
    field_bcs_[side]->fill_ghosts(u, static_cast<Side>(side));
  }
  // Every register write in this class is followed by a ghost refill of the
  // register it wrote (combine_stage, copy_state and its rollbacks, seed_state,
  // the positivity floors, and the CFL/divB preflights all do this), so hooking
  // invalidation here catches all of them from one place instead of relying on
  // each future mutation site to remember. A refill is itself a reason to
  // invalidate: reconstruction reads ghost cells, so their values are part of
  // its input.
  //
  // Only a refill of the CACHED register invalidates. Refilling a different one
  // -- copy_state(rk_[0], step_backup_) snapshotting the live state, for
  // instance -- cannot change what ifx_/ify_ describe, and invalidating on it
  // would destroy the cache before the stage that reuses it ever ran. This is
  // an address comparison, so it stays correct if registers are ever swapped
  // rather than copied. Logically const, like the ghost write it accompanies.
  if (interface_cache_source_ == &u) {
    const_cast<MhdSolver2D&>(*this).invalidate_interface_cache();
  }
}

BoundaryFlags4 MhdSolver2D::boundary_flags() const {
  // Per-side mode: periodic=0, outflow=1, wall=2, cylindrical axis=3,
  // exchanged internal tile side=4. Physical sides drop the ghost-gradient
  // dependence; periodic sides wrap and internal sides consume exchanged guards.
  // cfg_.boundary.field is ordered
  // [x_lo, x_hi, y_lo, y_hi], matching BoundaryFlags4::side. An all-periodic
  // deck yields all-zero flags (the periodic fast path).
  BoundaryFlags4 flags{};
  for (int side = 0; side < 4; ++side) {
    flags.side[side] = field_bcs_[side]->ghost_continuation_mode();
  }
  return flags;
}

void MhdSolver2D::compute_residual(const MhdField2D<Real>& u, MhdField2D<Real>& dudt) {
  compute_residual_flux_and_emf(u, dudt);
  finish_ct_emf();
  compute_ct_rate_from_emf(dudt);
  finish_split_energy(dudt);
}

void MhdSolver2D::compute_residual_flux_and_emf(
    const MhdField2D<Real>& u, MhdField2D<Real>& dudt) {
  prepare_residual_face_records(u, dudt);
  consume_residual_face_records(u, dudt);
}

void MhdSolver2D::prepare_residual_face_records(
    const MhdField2D<Real>& u, MhdField2D<Real>& dudt,
    FaceOwnershipFlags4 ownership) {
  const int order = positivity_reconstruction_order_ > 0
                        ? positivity_reconstruction_order_
                        : reconstruction_order();
  const int collocation_order = order <= 1 ? 1 : 0;
  // Public callers may reseed B0 component-by-component. Validate the completed
  // static field before any reconstruction, Riemann, CT, or source kernel reads
  // it; constructor-sampled and previously validated fields take the cached path.
  ensure_background_solenoidal();
  // Sample the cache BEFORE fill_ghosts, which invalidates unconditionally. In
  // the auto-dt loop cfl_limit() has just filled these same ghosts and
  // reconstructed this same register at this same order, so the refill below is
  // idempotent and the recorded states are still the ones reconstruction would
  // produce.
  const bool interfaces_current = interface_cache_valid(u, order);
  // Boundary fills may set physical high faces as well as ghost cells. Refresh
  // first so a live-state preflight validates exactly the collocation consumed
  // by reconstruction and the Riemann/CT kernels below.
  auto& mutable_u = const_cast<MhdField2D<Real>&>(u);
  fill_ghosts(mutable_u);
  if (&u == &rk_[0]) {
    ensure_live_state_solenoidal();
    ensure_live_state_admissible(collocation_order);
  }
  // The residual is built into dudt = L(u). Zero it, then accumulate -div F per
  // direction; cylindrical radial momentum is overwritten below by its fused
  // tensor-divergence form so the pressure-free curvature cancellation happens
  // before binary64 rounding.
  backend::device_memset_async(dudt.rho.device_ptr(), 0, dudt.rho.bytes(), nullptr);
  backend::device_memset_async(dudt.mx.device_ptr(), 0, dudt.mx.bytes(), nullptr);
  backend::device_memset_async(dudt.my.device_ptr(), 0, dudt.my.bytes(), nullptr);
  backend::device_memset_async(dudt.mz.device_ptr(), 0, dudt.mz.bytes(), nullptr);
  backend::device_memset_async(dudt.energy.device_ptr(), 0, dudt.energy.bytes(), nullptr);
  backend::device_memset_async(dudt.bx_face.device_ptr(), 0, dudt.bx_face.bytes(), nullptr);
  backend::device_memset_async(dudt.by_face.device_ptr(), 0, dudt.by_face.bytes(), nullptr);
  backend::device_memset_async(dudt.bz_cell.device_ptr(), 0, dudt.bz_cell.bytes(), nullptr);
  const Real gamma = cfg_.gamma;
  const BoundaryFlags4 flags = boundary_flags();

  // Reconstruct both directions and produce the complete HLLD face records,
  // including the factored active-background auxiliary channels. Distributed
  // stepping pauses after this phase so a single canonical owner can broadcast
  // each shared record before either adjacent tile consumes its divergence.
  if (!interfaces_current) {
    launch_mhd_reconstruct(u, b0_, 0, ifx_, order, flags, gamma, nullptr,
                           false, radial_tables_.view());
  }
  const bool low_order = order <= 1;
  launch_mhd_hlld_flux(
      ifx_, b0_, 0, flux_x_, flags, gamma, nullptr, low_order,
      b0_.active ? &momentum_flux_x_ : nullptr, order, ownership,
      radial_tables_.view());

  if (!interfaces_current) {
    launch_mhd_reconstruct(u, b0_, 1, ify_, order, flags, gamma, nullptr,
                           false, radial_tables_.view());
  }
  launch_mhd_hlld_flux(
      ify_, b0_, 1, flux_y_, flags, gamma, nullptr, low_order,
      b0_.active ? &momentum_flux_y_ : nullptr, order, ownership,
      radial_tables_.view());
}

void MhdSolver2D::consume_residual_face_records(
    const MhdField2D<Real>& u, MhdField2D<Real>& dudt) {
  const int order = positivity_reconstruction_order_ > 0
                        ? positivity_reconstruction_order_
                        : reconstruction_order();
  const int collocation_order = order <= 1 ? 1 : 0;
  const BoundaryFlags4 flags = boundary_flags();
  const Real gamma = cfg_.gamma;

  // Accumulate -div(F) from the already-canonical face records. This writes
  // all eight slots, including bx_face/by_face, but CT overwrites those two
  // staggered rates below with the pure EMF curl. The cell-centred toroidal B
  // retains its physical flux difference. With active B0, energy is overwritten
  // later by one fused invariant assembled from both directional records and
  // the final CT rate.
  launch_mhd_flux_difference(
      flux_x_, 0, dudt, nullptr, is_cylindrical());
  launch_mhd_flux_difference(
      flux_y_, 1, dudt, nullptr, is_cylindrical());
  if (b0_.active) {
    launch_mhd_split_momentum_residual(
        b0_, flux_x_, momentum_flux_x_, flux_y_, momentum_flux_y_,
        dudt, flags, nullptr, is_cylindrical(), collocation_order, order,
        radial_tables_.view());
  }
  if (is_cylindrical()) {
    launch_mhd_cylindrical_radial_momentum_residual(
        u, b0_, flux_x_, flux_y_, dudt, flags, nullptr,
        gamma, collocation_order, order,
        b0_.active ? &momentum_flux_x_ : nullptr,
        b0_.active ? &momentum_flux_y_ : nullptr,
        radial_tables_.view());
  }

  // Build the corner EMF from the two-direction interface states (kinematic
  // E = -(v x B)). Then write the discrete curl of that EMF as a RATE into the
  // face-B slots of dudt (overwrite, no dt), discarding the flux-difference
  // contamination in bx_face/by_face. Because face B is now carried as an ordinary
  // residual component, combine_stage advances it through the SAME SSP-RK3 convex
  // combination as the other 7 components -- and since the curl stencil telescopes
  // discretely (div(curl) = 0) and a convex combination of div-free fields stays
  // div-free, div(B) is preserved at round-off through every stage. There is
  // therefore no separate direct curl-into-the-field step: applying the curl to
  // face B on top of the flux divergence would double-count it.
  const bool low_order = order <= 1;
  launch_mhd_ct_emf_prepare(u, b0_, ifx_, ify_, flags, emf_, gamma, nullptr,
                            order, low_order, radial_tables_.view());
}

void MhdSolver2D::finish_ct_emf() {
  const int order = positivity_reconstruction_order_ > 0
                        ? positivity_reconstruction_order_
                        : reconstruction_order();
  launch_mhd_ct_emf_finish(boundary_flags(), emf_, nullptr, order,
                           is_cylindrical(), radial_tables_.view());
}

void MhdSolver2D::compute_ct_rate_from_emf(MhdField2D<Real>& dudt) {
  launch_mhd_emf_curl_rate(emf_, dudt, grid_, nullptr, is_cylindrical());
}

void MhdSolver2D::finish_split_energy(MhdField2D<Real>& dudt) {
  const int order = positivity_reconstruction_order_ > 0
                        ? positivity_reconstruction_order_
                        : reconstruction_order();
  const int collocation_order = order <= 1 ? 1 : 0;
  const BoundaryFlags4 flags = boundary_flags();
  // Derive the CT-rate ghost closure before its order-matched cell collocation
  // near a physical boundary. Then overwrite the active-background energy with
  // the complete invariant in one common-exponent sum. Retaining flux_x/y until
  // here prevents O(B0^2) face/CT terms from erasing a finite direct survivor.
  if (b0_.active) {
    fill_ghosts(dudt);
  }
  launch_mhd_split_energy_residual(
      b0_, flux_x_, momentum_flux_x_, flux_y_, momentum_flux_y_, dudt,
      flags, nullptr, is_cylindrical(), collocation_order, order,
      radial_tables_.view());

  // The interface-state host accessors and the next stage's reads are correct
  // only after the queued kernels finish; block once here so the seam is
  // sequentially consistent for the integrator.
  backend::device_synchronize(nullptr);
}

void MhdSolver2D::combine_stage(int stage, Real dt) {
  const Real theta = combine_stage_fraction(stage, dt);
  if (!(theta >= Real{1})) {
    throw PositivityRetry{std::isfinite(theta) ? theta : Real{0}};
  }
}

Real MhdSolver2D::combine_stage_fraction(int stage, Real dt) {
  apply_stage_update(stage, dt);
  return stage_admissible_fraction(stage);
}

int MhdSolver2D::apply_stage_update(int stage, Real dt) {
  if (!internal_integrator_access_) {
    note_external_mutable_state_access();
  }
  // Standard SSP-RK3 (Shu-Osher), with rk_[0]=U^n (live), rk_[1]=U1, rk_[2]=U2:
  //   stage 0: U1   = 1*U^n + 0*U1 + dt*L(U^n)         -> out rk_[1]
  //   stage 1: U2   = 3/4 U^n + 1/4 U1 + 1/4 dt*L(U1)  -> out rk_[2]
  //   stage 2: U^n+1= 1/3 U^n + 2/3 U2 + 2/3 dt*L(U2)  -> out rk_[0] (live)
  // launch_mhd_rk_stage(out, un, ustage, dudt, a, b, c_dt): out = a*un + b*ustage
  // + c_dt*dudt. The integrator calls compute_residual(stage input, residual_)
  // before each combine, so residual_ holds L of this stage's input field.
  MhdField2D<Real>* out = nullptr;
  Real a = Real{0}, b = Real{0}, c = Real{0};
  switch (stage) {
    case 0:
      out = &rk_[1];
      a = Real{1};
      b = Real{0};
      c = dt;
      break;
    case 1:
      out = &rk_[2];
      a = Real{3} / Real{4};
      b = Real{1} / Real{4};
      c = (Real{1} / Real{4}) * dt;
      break;
    case 2:
      out = &rk_[0];
      a = Real{1} / Real{3};
      b = Real{2} / Real{3};
      c = (Real{2} / Real{3}) * dt;
      break;
    default:
      throw std::invalid_argument{"MhdSolver2D::combine_stage: stage must be 0, 1, or 2"};
  }
  // For stage 0 the "ustage" operand is unused (b == 0); reuse rk_[1] as a safe
  // alias so we never dereference an uninitialized field. un is always U^n.
  const MhdField2D<Real>& un = rk_[0];
  const MhdField2D<Real>& ustage = (stage == 0) ? rk_[1] : (stage == 1) ? rk_[1] : rk_[2];

  // One uniform INTERIOR combine over all 8 components, INCLUDING the face-B slots.
  // Because
  // compute_residual already wrote the pure CT EMF-curl rate into residual_'s
  // bx_face/by_face slots (overwriting the flux-difference contamination), this
  // single convex combination advances face B by constrained transport alone --
  // there is deliberately no second, direct curl application. The Shu-Osher
  // weights stay consistent across every component, and div(B) is preserved at
  // round-off (convex combination of div-free fields + c*curl rate is div-free,
  // since div(curl) telescopes to zero discretely).
  launch_mhd_rk_stage(*out, un, ustage, residual_, a, b, c, nullptr);

  // Re-derive the ghost layers of the stage output from the configured BCs. This
  // is load-bearing for TWO reasons:
  //   1. launch_mhd_rk_stage intentionally updates interior cells/faces only;
  //      refilling derives every untouched ghost value from the new interior and
  //      keeps the boundary-ring divergence consistent with the configured BC.
  //   2. It leaves every stage output ghost-consistent, so the NEXT stage's
  //      reconstruction reads correct periodic-wrapped ghost CELLS and the
  //      boundary-face fluxes telescope (conserving mass/momentum/energy on a
  //      periodic grid). compute_residual still refills the input ghosts at its
  //      top; this makes the field consistent at stage completion too, so the
  //      final state() handed back after step() is ghost-consistent for any
  //      downstream reader (e.g. divergence_b_max).
  fill_ghosts(*out);

  return stage == 2 ? 0 : stage + 1;
}

Real MhdSolver2D::stage_admissible_fraction(int stage) {
  if (stage < 0 || stage >= kNumRkRegisters) {
    throw std::invalid_argument{
        "MhdSolver2D::stage_admissible_fraction: stage must be 0, 1, or 2"};
  }
  MhdField2D<Real>& out = rk_[stage == 2 ? 0 : stage + 1];

  // Conservative positivity control. The selected limiter computes a per-cell
  // convex admissible fraction relative to the saved start of this substep and
  // returns the global minimum.  Evolution uses the mathematical admissible set
  // rho>0 and internal energy>0.  A fixed positive density/pressure floor is not
  // an invariant set of the conservative MHD equations: at a state exactly on
  // such a floor, an outward physical flux can point outside it for every
  // positive dt. Configured floors therefore belong only to the explicit repair
  // API, not to automatic initialization or conservative time evolution. We
  // never clamp a cell or inject energy here: a non-positive candidate is
  // discarded and the whole conservative SSP-RK substep is retried at a smaller
  // CFL fraction by advance_positive().
  Real theta = Real{1};
  if (positivity_control_active_) {
    theta = positivity_->admissible_fraction(
        step_backup_, out, Real{0}, Real{0}, cfg_.gamma,
        positivity_reconstruction_order_, radial_tables_.view());
    if (stage == 2 && positivity_reconstruction_order_ == 0 &&
        positivity_low_order_anchor_available_) {
      // Preserve a usable low-order retry anchor across accepted configured-
      // order steps. MP and adjacent-face magnetic recovery are not ordered: a
      // state can have positive MP pressure but negative low-order pressure.
      // Allowing such a final state strands the next retry with no admissible
      // first-order base. Intermediate high-order stages need only their own EOS
      // because any rejection rolls all the way back to step_backup_.
      theta = std::min(theta, positivity_->admissible_fraction(
          step_backup_, out, Real{0}, Real{0}, cfg_.gamma,
          /*collocation_order=*/1, radial_tables_.view()));
    } else if (stage == 2 && positivity_reconstruction_order_ == 1) {
      // A completed fallback piece returns to the configured operator after the
      // requested interval. Keep its final state admissible under that operator;
      // low-order intermediate stages are intentionally judged only by the
      // adjacent-face EOS used to compute their residuals.
      theta = std::min(theta, positivity_->admissible_fraction(
          step_backup_, out, Real{0}, Real{0}, cfg_.gamma,
          /*collocation_order=*/0, radial_tables_.view()));
    }
  }
  return std::isfinite(theta) ? theta : Real{0};
}

void MhdSolver2D::copy_state(const MhdField2D<Real>& src, MhdField2D<Real>& dst) {
  // Reuse the componentwise RK launcher's exact (1,0,0) copy branch. That branch
  // does not evaluate the residual, which may contain NaN after a rejected
  // stage. Ghosts are derived state, so refill them from the copied interior
  // rather than carrying stale halo values between retries.
  launch_mhd_rk_stage(dst, src, src, residual_, Real{1}, Real{0}, Real{0}, nullptr);
  fill_ghosts(dst);
  backend::device_synchronize(nullptr);
}

void MhdSolver2D::advance_positive(Real dt) {
  // Per-piece rollback is insufficient once an earlier positivity substep has
  // already been accepted: a later device/integrator/controller exception must
  // not leave the public state advanced by an unreported fraction of `dt`.
  // Snapshot the whole request separately and restore it on every exceptional
  // exit. Preserve the last successful request's diagnostic as transactional
  // state too.
  const int previous_positivity_substeps = last_positivity_substeps_;
  const bool previous_live_state_solver_owned = live_state_solver_owned_;
  copy_state(rk_[0], request_backup_);
  try {
  // Every exit path, including a failure while taking/restoring a device
  // snapshot, must leave the public solver in its configured reconstruction
  // mode with no stale rollback state armed.
  struct ControllerStateReset {
    bool& active;
    bool& low_order_anchor_available;
    int& reconstruction_order;
    ~ControllerStateReset() {
      active = false;
      low_order_anchor_available = false;
      reconstruction_order = 0;
    }
  } reset{positivity_control_active_,
          positivity_low_order_anchor_available_,
          positivity_reconstruction_order_};

  Real remaining = dt;
  Real trial = dt;
  // Once a high-order candidate rejects an interval, finish that complete
  // interval with the low-order retry operator. Alternating back to MP after
  // each accepted low-order piece would repeatedly recreate the same
  // inadmissible candidate and makes the result depend on controller retries.
  bool low_order_interval = false;
  int retries = 0;
  int attempts = 0;
  constexpr int kMaxRetries = 80;
  constexpr int kMaxSubsteps = 100000;
  last_positivity_substeps_ = 0;
  positivity_low_order_anchor_available_ =
      positivity_->admissible_fraction(
          rk_[0], rk_[0], Real{0}, Real{0}, cfg_.gamma,
          /*collocation_order=*/1, radial_tables_.view()) >= Real{1};

  const auto no_progress_error = [&](const char* context, Real theta) {
    std::ostringstream message;
    message << std::setprecision(std::numeric_limits<Real>::max_digits10)
            << "MhdSolver2D: " << context
            << " cannot advance time (trial=" << trial
            << ", remaining=" << remaining
            << ", theta=" << theta
            << ", low_order=" << (low_order_interval ? "true" : "false")
            << ", retries=" << retries
            << ", accepted_substeps=" << last_positivity_substeps_
            << ", attempts=" << attempts << ')';
    return std::runtime_error{message.str()};
  };

  while (remaining > Real{0}) {
    if (++attempts > kMaxSubsteps) {
      throw std::runtime_error{
          "MhdSolver2D: positivity controller exceeded its substep limit"};
    }
    trial = std::min(trial, remaining);
    copy_state(rk_[0], step_backup_);
    positivity_control_active_ = true;
    try {
      struct InternalIntegratorAccess {
        bool& active;
        explicit InternalIntegratorAccess(bool& flag) : active{flag} {
          active = true;
        }
        ~InternalIntegratorAccess() { active = false; }
      } internal_access{internal_integrator_access_};
      integrator_->advance(*this, trial);
    } catch (const PositivityRetry& retry) {
      const bool high_order_attempt = positivity_reconstruction_order_ == 0;
      positivity_control_active_ = false;
      // Restore the configured reconstruction mode before any rollback work
      // that can itself throw, so an exceptional exit never leaves a latent
      // first-order override on the solver object.
      positivity_reconstruction_order_ = 0;
      copy_state(step_backup_, rk_[0]);
      if (++retries > kMaxRetries) {
        throw std::runtime_error{
            "MhdSolver2D: unable to find a positive conservative substep"};
      }
      if (high_order_attempt) {
        // First retry the same CFL-safe interval with the piecewise-constant
        // LF/HLL operator. Shrinking by a theta computed from the
        // rejected high-order candidate before changing operators can drive the
        // trial below representable time progress even though the first-order
        // HLL update is admissible at its own CFL. The two collocations are not
        // ordered for arbitrary face data, however: an MP-admissible live state
        // can in principle be inadmissible under an adjacent-face average. In
        // that exceptional case retain the high-order EOS and use theta to
        // subcycle instead of entering a low-order operator whose base state is
        // already outside its admissible set.
        const Real low_base_theta = positivity_->admissible_fraction(
            rk_[0], rk_[0], Real{0}, Real{0}, cfg_.gamma,
            /*collocation_order=*/1, radial_tables_.view());
        if (low_base_theta >= Real{1}) {
          positivity_low_order_anchor_available_ = true;
          low_order_interval = true;
          positivity_reconstruction_order_ = 1;
          trial = std::min(trial, cfl_limit_for_collocation(1));
          if (!(trial > Real{0}) || !std::isfinite(trial) ||
              !(remaining - trial < remaining)) {
            positivity_reconstruction_order_ = 0;
            throw no_progress_error("low-order CFL substep", low_base_theta);
          }
          continue;
        }
      }
      // Stay strictly inside the convex admissible bound. Clamp the reduction
      // to avoid both a nearly unchanged retry and catastrophic one-shot shrink;
      // repeated retries still converge when theta is extremely small.
      const Real suggested = Real{0.8} * retry.theta;
      const Real factor = std::max(Real{0.05}, std::min(Real{0.8}, suggested));
      trial *= factor;
      // A troubled low-order substep is retried with a smaller interval while
      // retaining the first-order Godunov spatial operator. If adjacent-face
      // collocation was not admissible at the live high-order state, this is a
      // high-order theta subcycle instead.
      positivity_reconstruction_order_ = low_order_interval ? 1 : 0;
      // Do not reject a positive substep merely because it is small relative to
      // the original request.  The only meaningful round-off test is whether it
      // can reduce the *current* residual interval in working precision.  This
      // also avoids silently dropping any representable part of that interval.
      if (!(trial > Real{0}) || !std::isfinite(trial) ||
          !(remaining - trial < remaining)) {
        positivity_reconstruction_order_ = 0;
        throw no_progress_error("positive conservative substep", retry.theta);
      }
      continue;
    } catch (...) {
      positivity_control_active_ = false;
      positivity_reconstruction_order_ = 0;
      copy_state(step_backup_, rk_[0]);
      throw;
    }
    positivity_control_active_ = false;
    live_state_solver_owned_ = !external_mutable_state_exposed_;

    retries = 0;
    ++last_positivity_substeps_;
    positivity_reconstruction_order_ = low_order_interval ? 1 : 0;
    if (trial == remaining) {
      // `trial` was clipped to the entire representable residual interval. Mark
      // it consumed exactly; never discard a merely-small positive remainder.
      remaining = Real{0};
      break;
    }
    const Real next_remaining = remaining - trial;
    if (!(next_remaining > Real{0}) || !(next_remaining < remaining)) {
      throw std::runtime_error{
          "MhdSolver2D: positivity substep made no finite time progress"};
    }
    remaining = next_remaining;

    // The accepted substep may have changed the signal speed. Grow cautiously,
    // but never exceed either the unadvanced interval or the fresh CFL bound.
    trial = std::min(
        std::min(Real{2} * trial, remaining),
        cfl_limit_for_collocation(low_order_interval ? 1 : 0));
  }
  } catch (...) {
    // Reset mode flags before restoration because copy/fill may itself report a
    // device error; no exceptional path may retain a latent low-order override.
    positivity_control_active_ = false;
    positivity_low_order_anchor_available_ = false;
    positivity_reconstruction_order_ = 0;
    last_positivity_substeps_ = previous_positivity_substeps;
    copy_state(request_backup_, rk_[0]);
    live_state_solver_owned_ =
        previous_live_state_solver_owned && !external_mutable_state_exposed_;
    throw;
  }
}

void MhdSolver2D::note_external_mutable_state_access() noexcept {
  external_mutable_state_exposed_ = true;
  live_state_solver_owned_ = false;
  // A writable reference just escaped, and the holder can change any register
  // without going through fill_ghosts. Assume it did: this is the conservative
  // direction, and these accessors are not on the hot path.
  invalidate_interface_cache();
}

MhdField2D<Real>& MhdSolver2D::state() noexcept {
  if (!internal_integrator_access_) {
    note_external_mutable_state_access();
  }
  return rk_[0];
}

MhdField2D<Real>& MhdSolver2D::rk_register(int k) {
  if (k < 0 || k >= kNumRkRegisters) {
    throw std::out_of_range{"MhdSolver2D::rk_register: index out of range"};
  }
  if (!internal_integrator_access_) {
    note_external_mutable_state_access();
  }
  return rk_[k];
}

MhdField2D<Real>& MhdSolver2D::residual_register() noexcept {
  if (!internal_integrator_access_) {
    note_external_mutable_state_access();
  }
  return residual_;
}

void MhdSolver2D::ensure_live_state_admissible(int collocation_order) const {
  // Testing the state against itself makes the limiter's convex-segment
  // reduction a strict, non-mutating preflight over every interior cell.  It
  // checks all eight components for finiteness and requires rho>0 and internal
  // energy (hence gas pressure)>0 with the exact EOS/collocation used by the
  // evolution kernels. This proof cannot be cached: state() and rk_register(0)
  // expose device buffers whose retained mutable handles can change the live
  // state at any later time, without passing through an invalidation hook.
  const Real theta = positivity_->admissible_fraction(
      rk_[0], rk_[0], Real{0}, Real{0}, cfg_.gamma,
      collocation_order, radial_tables_.view());
  if (!(theta >= Real{1})) {
    throw std::invalid_argument{
        "MhdSolver2D: live state must have finite components, rho > 0, "
        "and gas pressure > 0 in every interior cell"};
  }
}

void MhdSolver2D::ensure_live_state_solenoidal() const {
  // CT preserves the discrete divergence of the accepted seed; it does not
  // repair an inconsistent seed.  Reject a resolved defect before either the
  // Riemann/CFL path or the first residual consumes it.  The local diagnostic
  // is the ratio of global residual and directional-scale L-infinity norms.
  // Cartesian offsets cancel before either norm; the annular radial
  // contribution retains B_r/r curvature. The scaled representation is
  // independent of field units and binary exponent and remains meaningful at
  // local derivative nulls.
  Real relative_linf = Real{0};
  launch_mhd_ct_divb_relative_linf(
      rk_[0], divb_scratch_, &relative_linf, nullptr, is_cylindrical(),
      live_state_solver_owned_ && !external_mutable_state_exposed_);
  if (!(std::isfinite(relative_linf) &&
        relative_linf <= numerics::kDiscreteSolenoidalTolerance)) {
    Real absolute_linf = Real{0};
    launch_mhd_ct_divb_linf(
        rk_[0], divb_scratch_, &absolute_linf, nullptr, is_cylindrical());
    std::ostringstream message;
    message << "MhdSolver2D: live magnetic field is not discretely "
               "divergence-free (relative L-infinity defect "
            << std::setprecision(std::numeric_limits<Real>::max_digits10)
            << relative_linf << ", absolute L-infinity residual "
            << absolute_linf << ')';
    throw std::invalid_argument{message.str()};
  }
}

Real MhdSolver2D::cfl_limit() const {
  return cfl_limit_for_collocation(/*collocation_order=*/0);
}

Real MhdSolver2D::cfl_limit_for_collocation(int collocation_order) const {
  // Reduce the maximum interior ADDITIVE incident-face signal rate on device,
  // then return cfl / max_rate. Configured HLLD uses the absolute outer-fan
  // bound max_side|v_n|+max_side(c_fast,n) at each face; the piecewise-constant
  // positivity retry uses its actual LF alpha=max_side(|v_n|+c_fast,n). The
  // additive (rather than per-direction max over min(dx,dy)) cell bound is the
  // correct stability limit for the UNSPLIT residual, which sums both flux
  // differences into one dudt per SSP-RK3 stage. The reduction is b0-aware (the
  // fast speed sees B=B0+b) and reads the exact reconstructed L/R states; an
  // invalid face contributes infinity. Cylindrical mode applies the exact
  // radial operator self-weights with (dr,dz)=(dx,dy).
  //
  ensure_background_solenoidal();
  // Incident-face CFL states include one derived ghost cell at physical or
  // periodic boundary faces. Refresh those values before BOTH admissibility
  // and the face reduction so the preflight checks exactly the collocated state
  // the CFL and residual paths consume (field BCs may also set physical high
  // faces). This is logically const because those slots are boundary-derived.
  auto& self = const_cast<MhdSolver2D&>(*this);
  self.fill_ghosts(self.rk_[0]);
  // Boundary launches and the admissibility reduction share the default stream,
  // so stream ordering makes the refreshed state visible before the host result
  // is returned by ensure_live_state_admissible().
  ensure_live_state_solenoidal();
  ensure_live_state_admissible(collocation_order);
  const int order = collocation_order > 0 ? 1 : reconstruction_order();
  const BoundaryFlags4 flags = boundary_flags();
  launch_mhd_reconstruct(
      self.rk_[0], b0_, 0, self.ifx_, order, flags, cfg_.gamma, nullptr,
      /*rate_only=*/true, radial_tables_.view());
  launch_mhd_reconstruct(
      self.rk_[0], b0_, 1, self.ify_, order, flags, cfg_.gamma, nullptr,
      /*rate_only=*/true, radial_tables_.view());
  // ifx_/ify_ now hold the reconstruction of the live register at `order`. In
  // the auto-dt loop the very next thing that happens is compute_residual on
  // this same unchanged register, which would recompute exactly this. Record
  // the cache AFTER the fill_ghosts above (which invalidates) so the recorded
  // generation is the post-fill one the reconstruction actually read.
  //
  // rate_only=true is recorded as equivalent to the residual path's
  // rate_only=false because the flag's only effect is choosing
  // reconstructed_rate_state_admissible over reconstructed_state_admissible,
  // and the former forwards directly to the latter. If those two predicates
  // ever diverge, this cache must start distinguishing them.
  self.note_interface_cache(self.rk_[0], order);
  ScaledCflRate max_rate{};
  launch_mhd_cfl_max_rate(
      self.ifx_, self.ify_, b0_, cfg_.gamma, cfl_scratch_,
      &max_rate, nullptr, order, is_cylindrical(), flags,
      radial_tables_.view());

  // The Suresh-Huynh MP limiter uses alpha=4, whose forward-Euler
  // monotonicity bound is nu <= 1/(1+alpha)=0.2. SSP-RK3 inherits that
  // coefficient. MUSCL and the piecewise-constant positivity retry retain the
  // user-selected factor.
  const Real effective_cfl = order >= 5
      ? std::min(cfg_.cfl, Real{0.2}) : cfg_.cfl;

  if (!(std::isfinite(max_rate.scaled_max_rate) &&
        std::isfinite(max_rate.rate_scale) && max_rate.rate_scale > Real{0})) {
    throw std::runtime_error{"MhdSolver2D: non-finite state encountered in CFL reduction"};
  }
  if (!(max_rate.scaled_max_rate > Real{0})) {
    // Reached only when every interior cell has a non-positive / non-finite
    // signal rate -- e.g. a pressureless, field-free, motionless state (the
    // positivity controller keeps rho>0 and p>0, so a normal run always yields a
    // finite positive rate). Any dt is stable for a truly signal-free field;
    // return a finite, dimensionally-consistent dt using the additive rate with
    // a unit reference speed (cfl / (1/dx + 1/dy)) so the fallback is a TIME like
    // the main path, not a length.
    const Real h_min = std::min(grid_.dx(), grid_.dy());
    const Real h_max = std::max(grid_.dx(), grid_.dy());
    const Real fallback =
        effective_cfl * (h_min / (Real{1} + h_min / h_max));
    if (!(fallback > Real{0}) || !std::isfinite(fallback)) {
      throw std::runtime_error{
          "MhdSolver2D: no finite positive CFL timestep is representable"};
    }
    return fallback;
  }
  // Keep the ordinary final division bit-for-bit unchanged. The overflow retry
  // scales every face/metric contribution homogeneously, so cfl/rate =
  // (cfl*rate_scale)/scaled_max_rate without materializing the true
  // out-of-range aggregate rate.
  const Real result = max_rate.rate_scale == Real{1}
      ? effective_cfl / max_rate.scaled_max_rate
      : (effective_cfl * max_rate.rate_scale) /
            max_rate.scaled_max_rate;
  if (!(result > Real{0}) || !std::isfinite(result)) {
    throw std::runtime_error{
        "MhdSolver2D: no finite positive CFL timestep is representable"};
  }
  return result;
}

Real MhdSolver2D::divergence_b_max() const {
  // The cell-centered divB stencil at the last interior column/row reads a GHOST
  // face (bx_face(i+1,j) / by_face(i,j+1)). Refill the field ghosts from the
  // configured BC first so those ghost faces match the current interior under the
  // periodic wrap (or other closure); measuring a ghost-stale field would
  // otherwise report a spurious nonzero div concentrated in the boundary ring,
  // even when the interior is exactly div-free. Logically const (only ghost
  // layers are touched, and they are derived state), hence the const_cast.
  ensure_background_solenoidal();
  auto& self = const_cast<MhdSolver2D&>(*this);
  self.fill_ghosts(self.rk_[0]);
  backend::device_synchronize(nullptr);
  Real linf = Real{0};
  launch_mhd_ct_divb_linf(self.rk_[0], divb_scratch_, &linf, nullptr,
                          is_cylindrical());
  return linf;
}

void MhdSolver2D::check_cfl(Real dt) const {
  if (!(dt > Real{0}) || !std::isfinite(dt)) {
    throw std::invalid_argument{"MhdSolver2D: dt must be finite and positive"};
  }
  if (dt > cfl_limit()) {
    throw std::invalid_argument{
        "MhdSolver2D: dt exceeds the CFL stability limit for this grid and scheme"};
  }
}

void MhdSolver2D::step(Real dt) {
  check_cfl(dt);
  advance_positive(dt);
}

void MhdSolver2D::step_unchecked(Real dt) {
  // CFL pre-validated by the caller (the auto-dt loop just called cfl_limit());
  // skip the redundant full-grid reduction inside check_cfl(). Still guard the
  // sign so a bad caller cannot advance with a non-positive dt.
  if (!(dt > Real{0}) || !std::isfinite(dt)) {
    throw std::invalid_argument{"MhdSolver2D: dt must be finite and positive"};
  }
  ensure_background_solenoidal();
  ensure_live_state_admissible();
  advance_positive(dt);
}

void MhdSolver2D::advance(Real t_end, Real dt) {
  if (!(t_end >= Real{0}) || !std::isfinite(t_end)) {
    throw std::invalid_argument{
        "MhdSolver2D::advance: t_end must be finite and non-negative"};
  }
  if (!(dt > Real{0}) || !std::isfinite(dt)) {
    throw std::invalid_argument{
        "MhdSolver2D::advance: dt must be finite and positive"};
  }
  Real t = Real{0};
  while (t < t_end) {
    const Real remaining = t_end - t;
    const Real dt_step = std::min(dt, remaining);
    if (!(dt_step > Real{0}) || !std::isfinite(dt_step)) {
      throw std::runtime_error{
          "MhdSolver2D::advance: timestep made no finite forward progress"};
    }
    step(dt_step);
    if (dt_step == remaining) {
      // The clipped last step lands on the requested endpoint by construction;
      // assign it exactly instead of relying on t + (t_end - t) rounding.
      t = t_end;
    } else {
      const Real next = t + dt_step;
      if (!(next > t) || !std::isfinite(next)) {
        throw std::runtime_error{
            "MhdSolver2D::advance: timestep made no finite forward progress"};
      }
      t = next;
    }
  }
}

}  // namespace quasar::mhd
