// Analytic equilibrium/profile handoff for the toroidal ideal-MHD operator.

#include "quasar/physics/stability/toroidal_equilibrium.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace quasar::stability {
namespace {

inline constexpr unsigned kBlock1D = 128;
inline constexpr unsigned kBlock2D = 16;

enum class BuildStatus : int {
  success = 0,
  invalid_lambda = 1,
  nonpositive_f_squared = 2,
  nonpositive_density = 3,
  nonfinite_profile = 4,
  invalid_geometry = 5,
};

struct GridView {
  int nr;
  int nz;
  Real r_min;
  Real z_min;
  Real dr;
  Real dz;

  __device__ std::size_t index(int i, int j) const {
    return static_cast<std::size_t>(i)
         + static_cast<std::size_t>(nr) * j;
  }
};

GridView view_of(const numerics::EllipticGrid& grid) {
  return GridView{grid.nr, grid.nz, grid.r_min, grid.z_min, grid.dr(),
                  grid.dz()};
}

std::size_t checked_product(std::size_t a, std::size_t b,
                            const char* message) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::length_error{message};
  }
  return a * b;
}

template <class T>
void require_size(const backend::DeviceBuffer<T>& buffer,
                  std::size_t expected, const char* name) {
  if (buffer.size() != expected) {
    throw std::invalid_argument{std::string{name} + " has size "
                                + std::to_string(buffer.size())
                                + "; expected " + std::to_string(expected)};
  }
}

template <class T>
void require_owner(const backend::DeviceBuffer<T>& buffer, int expected,
                   const char* name) {
  if (buffer.owner_device() != expected) {
    throw std::invalid_argument{std::string{name}
                                + " is allocated on a different device"};
  }
}

struct CoordinateShape {
  int n_lambda{0};
  int n_theta{0};
  int owner{-1};
  std::size_t radial_points{0};
  std::size_t geometry_points{0};
};

CoordinateShape validate_coordinates(const FluxCoordinateGrid& coords) {
  if (coords.n_psi < 1 || coords.n_theta < 1) {
    throw std::invalid_argument{
        "toroidal equilibrium requires a non-empty FluxCoordinateGrid"};
  }
  const std::size_t radial_points = static_cast<std::size_t>(coords.n_psi);
  const std::size_t geometry_points = checked_product(
      radial_points, static_cast<std::size_t>(coords.n_theta),
      "toroidal equilibrium geometry size overflows size_t");

  require_size(coords.psi_n, radial_points, "FluxCoordinateGrid::psi_n");
  require_size(coords.q, radial_points, "FluxCoordinateGrid::q");
  require_size(coords.valid, radial_points, "FluxCoordinateGrid::valid");
  require_size(coords.r, geometry_points, "FluxCoordinateGrid::r");
  require_size(coords.z, geometry_points, "FluxCoordinateGrid::z");
  require_size(coords.dr_dpsi, geometry_points,
               "FluxCoordinateGrid::dr_dpsi");
  require_size(coords.dz_dpsi, geometry_points,
               "FluxCoordinateGrid::dz_dpsi");
  require_size(coords.dr_dtheta, geometry_points,
               "FluxCoordinateGrid::dr_dtheta");
  require_size(coords.dz_dtheta, geometry_points,
               "FluxCoordinateGrid::dz_dtheta");
  require_size(coords.jacobian, geometry_points,
               "FluxCoordinateGrid::jacobian");
  require_size(coords.g_psipsi, geometry_points,
               "FluxCoordinateGrid::g_psipsi");
  require_size(coords.g_psitheta, geometry_points,
               "FluxCoordinateGrid::g_psitheta");
  require_size(coords.g_thetatheta, geometry_points,
               "FluxCoordinateGrid::g_thetatheta");

  const int owner = coords.psi_n.owner_device();
  if (owner < 0) {
    throw std::invalid_argument{
        "toroidal equilibrium coordinates are not device-resident"};
  }
  require_owner(coords.q, owner, "FluxCoordinateGrid::q");
  require_owner(coords.valid, owner, "FluxCoordinateGrid::valid");
  require_owner(coords.r, owner, "FluxCoordinateGrid::r");
  require_owner(coords.z, owner, "FluxCoordinateGrid::z");
  require_owner(coords.dr_dpsi, owner, "FluxCoordinateGrid::dr_dpsi");
  require_owner(coords.dz_dpsi, owner, "FluxCoordinateGrid::dz_dpsi");
  require_owner(coords.dr_dtheta, owner, "FluxCoordinateGrid::dr_dtheta");
  require_owner(coords.dz_dtheta, owner, "FluxCoordinateGrid::dz_dtheta");
  require_owner(coords.jacobian, owner, "FluxCoordinateGrid::jacobian");
  require_owner(coords.g_psipsi, owner, "FluxCoordinateGrid::g_psipsi");
  require_owner(coords.g_psitheta, owner,
                "FluxCoordinateGrid::g_psitheta");
  require_owner(coords.g_thetatheta, owner,
                "FluxCoordinateGrid::g_thetatheta");
  if (backend::current_device() != owner) {
    throw std::invalid_argument{
        "toroidal equilibrium coordinates are not on the current device"};
  }

  return CoordinateShape{coords.n_psi, coords.n_theta, owner, radial_points,
                         geometry_points};
}

void validate_profile_metadata(
    const equilibrium::ProfileCoefficients& profile,
    const DensityProfileCoefficients& density, Real psi_axis,
    Real psi_boundary, Real profile_scale, Real f_vacuum) {
  if (profile.n_p < 1
      || profile.n_p > equilibrium::ProfileCoefficients::kMaxCoefficients
      || profile.n_f < 1
      || profile.n_f > equilibrium::ProfileCoefficients::kMaxCoefficients) {
    throw std::invalid_argument{
        "toroidal equilibrium requires non-empty polynomial P and FF' "
        "profiles within ProfileCoefficients capacity"};
  }
  if (density.count < 1
      || density.count > DensityProfileCoefficients::kMaxCoefficients) {
    throw std::invalid_argument{
        "toroidal equilibrium requires a non-empty polynomial density "
        "profile within DensityProfileCoefficients capacity"};
  }
  if (!std::isfinite(psi_axis) || !std::isfinite(psi_boundary)
      || !std::isfinite(profile_scale) || !std::isfinite(f_vacuum)
      || psi_axis == psi_boundary) {
    throw std::invalid_argument{
        "toroidal equilibrium scalar inputs must be finite and have "
        "psi_axis != psi_boundary"};
  }
}

void validate_equilibrium_output(const ToroidalEquilibriumFields& out,
                                 const CoordinateShape& shape) {
  if (out.n_lambda != shape.n_lambda || out.n_theta != shape.n_theta) {
    throw std::invalid_argument{
        "ToroidalEquilibriumFields shape must match FluxCoordinateGrid"};
  }
#define QUASAR_REQUIRE_RADIAL(member)                                           \
  require_size(out.member, shape.radial_points,                                \
               "ToroidalEquilibriumFields::" #member);                       \
  require_owner(out.member, shape.owner,                                       \
                "ToroidalEquilibriumFields::" #member)
#define QUASAR_REQUIRE_GEOMETRY(member)                                         \
  require_size(out.member, shape.geometry_points,                              \
               "ToroidalEquilibriumFields::" #member);                       \
  require_owner(out.member, shape.owner,                                       \
                "ToroidalEquilibriumFields::" #member)

  QUASAR_REQUIRE_RADIAL(pressure);
  QUASAR_REQUIRE_RADIAL(pressure_lambda);
  QUASAR_REQUIRE_RADIAL(f);
  QUASAR_REQUIRE_RADIAL(f_squared);
  QUASAR_REQUIRE_RADIAL(f_lambda);
  QUASAR_REQUIRE_RADIAL(ff_lambda);
  QUASAR_REQUIRE_RADIAL(density);
  QUASAR_REQUIRE_GEOMETRY(g_lambda_lambda);
  QUASAR_REQUIRE_GEOMETRY(g_lambda_theta);
  QUASAR_REQUIRE_GEOMETRY(g_theta_theta);
  QUASAR_REQUIRE_GEOMETRY(g_phi_phi);
  QUASAR_REQUIRE_GEOMETRY(jacobian);
  QUASAR_REQUIRE_GEOMETRY(b_theta);
  QUASAR_REQUIRE_GEOMETRY(b_phi);
  QUASAR_REQUIRE_GEOMETRY(j_theta);
  QUASAR_REQUIRE_GEOMETRY(j_phi);

#undef QUASAR_REQUIRE_GEOMETRY
#undef QUASAR_REQUIRE_RADIAL
}

__device__ inline Real evaluate_polynomial(const Real* coefficients, int count,
                                           Real x) {
  Real value = Real{0};
  for (int k = count; k-- > 0;) value = value * x + coefficients[k];
  return value;
}

// Integral from x to 1 of a polynomial, evaluated through its analytic
// antiderivative.  The operation order is fixed and x=1 cancels exactly.
__device__ inline Real integrate_polynomial_to_one(const Real* coefficients,
                                                   int count, Real x) {
  Real at_one = Real{0};
  Real antiderivative_polynomial = Real{0};
  for (int k = count; k-- > 0;) {
    const Real coefficient = coefficients[k] / static_cast<Real>(k + 1);
    at_one += coefficient;
    antiderivative_polynomial = antiderivative_polynomial * x + coefficient;
  }
  return at_one - x * antiderivative_polynomial;
}

__global__ void evaluate_profiles_kernel(
    const Real* __restrict__ lambda, int n_lambda,
    equilibrium::ProfileCoefficients profile,
    DensityProfileCoefficients density_profile, Real delta_psi,
    Real profile_scale, Real f_vacuum, Real* __restrict__ pressure,
    Real* __restrict__ pressure_lambda, Real* __restrict__ f,
    Real* __restrict__ f_squared, Real* __restrict__ f_lambda,
    Real* __restrict__ ff_lambda, Real* __restrict__ density) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n_lambda) return;

  const Real x = lambda[i];
  const Real p_prime =
      evaluate_polynomial(profile.p_coeffs, profile.n_p, x);
  const Real ff_prime =
      evaluate_polynomial(profile.f_coeffs, profile.n_f, x);
  const Real p_integral =
      integrate_polynomial_to_one(profile.p_coeffs, profile.n_p, x);
  const Real ff_integral =
      integrate_polynomial_to_one(profile.f_coeffs, profile.n_f, x);
  const Real rho = evaluate_polynomial(
      density_profile.coefficients, density_profile.count, x);

  const Real scaled_delta = delta_psi * profile_scale;
  const Real p_lambda = scaled_delta * p_prime;
  const Real ff_l = scaled_delta * ff_prime;
  const Real f2 = f_vacuum * f_vacuum
                - Real{2} * scaled_delta * ff_integral;

  pressure[i] = -scaled_delta * p_integral;
  pressure_lambda[i] = p_lambda;
  f_squared[i] = f2;
  ff_lambda[i] = ff_l;
  density[i] = rho;

  if (f2 > Real{0} && isfinite(f2)) {
    const Real f_value = copysign(sqrt(f2), f_vacuum);
    f[i] = f_value;
    f_lambda[i] = ff_l / f_value;
  } else {
    f[i] = Real{0};
    f_lambda[i] = Real{0};
  }
}

__global__ void build_geometry_kernel(
    const Real* __restrict__ r, const Real* __restrict__ r_lambda,
    const Real* __restrict__ z_lambda,
    const Real* __restrict__ r_theta_stored,
    const Real* __restrict__ z_theta_stored,
    const Real* __restrict__ q, const Real* __restrict__ p_lambda,
    const Real* __restrict__ f_lambda, int n_lambda, int n_theta,
    Real signed_flux_scale, Real* __restrict__ g_lambda_lambda,
    Real* __restrict__ g_lambda_theta,
    Real* __restrict__ g_theta_theta, Real* __restrict__ g_phi_phi,
    Real* __restrict__ jacobian, Real* __restrict__ b_theta,
    Real* __restrict__ b_phi, Real* __restrict__ j_theta,
    Real* __restrict__ j_phi) {
  const int theta = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int radial = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (radial >= n_lambda || theta >= n_theta) return;

  const std::size_t k = static_cast<std::size_t>(radial) * n_theta + theta;
  const Real radius = r[k];
  const Real rl = r_lambda[k];
  const Real zl = z_lambda[k];
  const Real rt = r_theta_stored[k];
  const Real zt = z_theta_stored[k];

  const Real det_stored = rl * zt - rt * zl;
  const Real j = radius * det_stored;
  const Real bth = signed_flux_scale / j;
  const Real jth = -f_lambda[radial] / (mu0 * j);

  g_lambda_lambda[k] = rl * rl + zl * zl;
  // theta_source = -theta_stored: only the mixed covariant metric flips.
  g_lambda_theta[k] = -(rl * rt + zl * zt);
  g_theta_theta[k] = rt * rt + zt * zt;
  g_phi_phi[k] = radius * radius;
  jacobian[k] = j;
  b_theta[k] = bth;
  b_phi[k] = q[radial] * bth;
  j_theta[k] = jth;
  j_phi[k] = q[radial] * jth - p_lambda[radial] / signed_flux_scale;
}

__global__ void validate_build_kernel(
    const Real* __restrict__ lambda, const Real* __restrict__ q,
    const int* __restrict__ valid, int n_lambda, int n_theta,
    const Real* __restrict__ r, const Real* __restrict__ r_lambda,
    const Real* __restrict__ z_lambda,
    const Real* __restrict__ r_theta_stored,
    const Real* __restrict__ z_theta_stored,
    const Real* __restrict__ stored_jacobian,
    const Real* __restrict__ pressure,
    const Real* __restrict__ pressure_lambda, const Real* __restrict__ f,
    const Real* __restrict__ f_squared,
    const Real* __restrict__ f_lambda,
    const Real* __restrict__ ff_lambda,
    const Real* __restrict__ density,
    const Real* __restrict__ g_lambda_lambda,
    const Real* __restrict__ g_lambda_theta,
    const Real* __restrict__ g_theta_theta,
    const Real* __restrict__ g_phi_phi,
    const Real* __restrict__ jacobian,
    const Real* __restrict__ b_theta,
    const Real* __restrict__ b_phi,
    const Real* __restrict__ j_theta,
    const Real* __restrict__ j_phi, int* __restrict__ status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  *status = static_cast<int>(BuildStatus::success);

  for (int radial = 0; radial < n_lambda; ++radial) {
    const Real x = lambda[radial];
    if (valid[radial] != 1 || !isfinite(x) || x < Real{0}
        || x > Real{1} || !isfinite(q[radial])) {
      *status = static_cast<int>(BuildStatus::invalid_lambda);
      return;
    }
  }
  for (int radial = 0; radial < n_lambda; ++radial) {
    if (!isfinite(pressure[radial]) || !isfinite(pressure_lambda[radial])
        || !isfinite(f[radial]) || !isfinite(f_squared[radial])
        || !isfinite(f_lambda[radial]) || !isfinite(ff_lambda[radial])
        || !isfinite(density[radial])) {
      *status = static_cast<int>(BuildStatus::nonfinite_profile);
      return;
    }
  }
  for (int radial = 0; radial < n_lambda; ++radial) {
    if (!(f_squared[radial] > Real{0})) {
      *status = static_cast<int>(BuildStatus::nonpositive_f_squared);
      return;
    }
  }
  for (int radial = 0; radial < n_lambda; ++radial) {
    if (!(density[radial] > Real{0})) {
      *status = static_cast<int>(BuildStatus::nonpositive_density);
      return;
    }
  }

  constexpr Real tolerance = Real{64} * std::numeric_limits<Real>::epsilon();
  for (int radial = 0; radial < n_lambda; ++radial) {
    for (int theta = 0; theta < n_theta; ++theta) {
      const std::size_t k =
          static_cast<std::size_t>(radial) * n_theta + theta;
      const Real radius = r[k];
      const Real rl = r_lambda[k];
      const Real zl = z_lambda[k];
      const Real rt = r_theta_stored[k];
      const Real zt = z_theta_stored[k];
      const Real direct_j = radius * (rl * zt - rt * zl);
      const Real scale = fmax(Real{1}, fabs(direct_j));
      if (!(radius > Real{0}) || direct_j == Real{0}
          || !isfinite(radius) || !isfinite(rl) || !isfinite(zl)
          || !isfinite(rt) || !isfinite(zt) || !isfinite(direct_j)
          || !isfinite(stored_jacobian[k])
          || fabs(stored_jacobian[k] - direct_j) > tolerance * scale
          || !isfinite(g_lambda_lambda[k])
          || !isfinite(g_lambda_theta[k])
          || !isfinite(g_theta_theta[k]) || !isfinite(g_phi_phi[k])
          || !isfinite(jacobian[k]) || !isfinite(b_theta[k])
          || !isfinite(b_phi[k]) || !isfinite(j_theta[k])
          || !isfinite(j_phi[k])) {
        *status = static_cast<int>(BuildStatus::invalid_geometry);
        return;
      }
    }
  }
}

__device__ inline void cubic_lagrange_weights(Real x, Real* weights) {
  weights[0] = -(x - Real{1}) * (x - Real{2}) * (x - Real{3}) / Real{6};
  weights[1] = x * (x - Real{2}) * (x - Real{3}) / Real{2};
  weights[2] = -x * (x - Real{1}) * (x - Real{3}) / Real{2};
  weights[3] = x * (x - Real{1}) * (x - Real{2}) / Real{6};
}

__device__ inline Real sample_field_cubic(const GridView& grid,
                                          const Real* field, Real r,
                                          Real z) {
  if (!isfinite(r) || !isfinite(z)) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  Real fi = (r - grid.r_min) / grid.dr;
  Real fj = (z - grid.z_min) / grid.dz;
  if (!isfinite(fi) || !isfinite(fj)) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  fi = fmin(fmax(fi, Real{0}), static_cast<Real>(grid.nr - 1));
  fj = fmin(fmax(fj, Real{0}), static_cast<Real>(grid.nz - 1));

  int i0 = static_cast<int>(floor(fi)) - 1;
  int j0 = static_cast<int>(floor(fj)) - 1;
  i0 = max(0, min(i0, grid.nr - 4));
  j0 = max(0, min(j0, grid.nz - 4));

  Real wr[4];
  Real wz[4];
  cubic_lagrange_weights(fi - static_cast<Real>(i0), wr);
  cubic_lagrange_weights(fj - static_cast<Real>(j0), wz);

  Real value = Real{0};
  for (int jj = 0; jj < 4; ++jj) {
    Real row = Real{0};
    for (int ii = 0; ii < 4; ++ii) {
      row += wr[ii] * field[grid.index(i0 + ii, j0 + jj)];
    }
    value += wz[jj] * row;
  }
  return value;
}

__device__ inline Real finite_relative_deviation(Real value, Real reference) {
  constexpr Real maximum = std::numeric_limits<Real>::max();
  constexpr Real minimum = std::numeric_limits<Real>::min();
  if (!isfinite(value) || !isfinite(reference)) return maximum;
  const Real denominator = fabs(reference);
  if (!(denominator > minimum)) return value == reference ? Real{0} : maximum;
  const Real deviation = fabs(value - reference) / denominator;
  return isfinite(deviation) ? deviation : maximum;
}

__global__ void validate_geometry_kernel(
    GridView grid, const Real* __restrict__ r,
    const Real* __restrict__ z, const Real* __restrict__ r_lambda,
    const Real* __restrict__ z_lambda,
    const Real* __restrict__ r_theta_stored,
    const Real* __restrict__ z_theta_stored,
    const Real* __restrict__ q, const int* __restrict__ valid, int n_lambda,
    int n_theta, const Real* __restrict__ field_br,
    const Real* __restrict__ field_bz,
    const Real* __restrict__ field_bphi, Real signed_flux_scale,
    Real* __restrict__ q_deviation, Real* __restrict__ s_deviation,
    Real* __restrict__ pest_deviation) {
  const int radial = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (radial >= n_lambda) return;

  constexpr Real maximum = std::numeric_limits<Real>::max();
  if (valid[radial] != 1) {
    q_deviation[radial] = maximum;
    s_deviation[radial] = maximum;
    pest_deviation[radial] = maximum;
    return;
  }

  const std::size_t base = static_cast<std::size_t>(radial) * n_theta;
  Real pest_sum = Real{0};
  bool malformed = false;
  for (int theta = 0; theta < n_theta; ++theta) {
    const std::size_t k = base + theta;
    const Real radius = r[k];
    const Real det_stored = r_lambda[k] * z_theta_stored[k]
                          - r_theta_stored[k] * z_lambda[k];
    const Real j_source = radius * det_stored;
    const Real value = j_source / (radius * radius);
    if (!(radius > Real{0}) || det_stored == Real{0}
        || !isfinite(value)) {
      malformed = true;
      break;
    }
    pest_sum += value;
  }
  if (malformed || !isfinite(pest_sum)) {
    q_deviation[radial] = maximum;
    s_deviation[radial] = maximum;
    pest_deviation[radial] = maximum;
    return;
  }

  const Real pest_mean = pest_sum / static_cast<Real>(n_theta);
  Real worst_q = Real{0};
  Real worst_s = Real{0};
  Real worst_pest = Real{0};
  for (int theta = 0; theta < n_theta; ++theta) {
    const std::size_t k = base + theta;
    const Real radius = r[k];
    const Real rl = r_lambda[k];
    const Real zl = z_lambda[k];
    const Real rt = r_theta_stored[k];
    const Real zt = z_theta_stored[k];
    const Real det_stored = rl * zt - rt * zl;
    const Real j_source = radius * det_stored;

    const Real br = sample_field_cubic(grid, field_br, radius, z[k]);
    const Real bz = sample_field_cubic(grid, field_bz, radius, z[k]);
    const Real bphi_physical =
        sample_field_cubic(grid, field_bphi, radius, z[k]);
    const Real btheta_source = (br * zl - bz * rl) / det_stored;
    const Real bphi_contravariant = bphi_physical / radius;
    const Real q_from_field = bphi_contravariant / btheta_source;
    const Real s_from_field = j_source * btheta_source;
    const Real pest_value = j_source / (radius * radius);

    worst_q = fmax(worst_q,
                   finite_relative_deviation(q_from_field, q[radial]));
    worst_s = fmax(worst_s, finite_relative_deviation(
                                     s_from_field, signed_flux_scale));
    worst_pest = fmax(worst_pest,
                      finite_relative_deviation(pest_value, pest_mean));
  }

  q_deviation[radial] = isfinite(worst_q) ? worst_q : maximum;
  s_deviation[radial] = isfinite(worst_s) ? worst_s : maximum;
  pest_deviation[radial] = isfinite(worst_pest) ? worst_pest : maximum;
}

__device__ bool valid_geometry_deviation(Real value) {
  return isfinite(value) && value >= Real{0}
      && value < std::numeric_limits<Real>::max();
}

__global__ void summarize_geometry_validation_kernel(
    const Real* __restrict__ q_deviation,
    const Real* __restrict__ flux_scale_deviation,
    const Real* __restrict__ pest_deviation, int n_surfaces,
    ToroidalGeometryValidationSummary* __restrict__ out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  ToroidalGeometryValidationSummary summary{};
  summary.first_invalid_surface = -1;
  for (int surface = 0; surface < n_surfaces; ++surface) {
    const Real q_value = q_deviation[surface];
    const Real flux_scale_value = flux_scale_deviation[surface];
    const Real pest_value = pest_deviation[surface];
    const bool q_valid = valid_geometry_deviation(q_value);
    const bool flux_scale_valid = valid_geometry_deviation(flux_scale_value);
    const bool pest_valid = valid_geometry_deviation(pest_value);

    if (q_valid) {
      summary.maximum_q_relative_deviation =
          fmax(summary.maximum_q_relative_deviation, q_value);
    }
    if (flux_scale_valid) {
      summary.maximum_flux_scale_relative_deviation =
          fmax(summary.maximum_flux_scale_relative_deviation,
               flux_scale_value);
    }
    if (pest_valid) {
      summary.maximum_pest_j_over_r2_relative_deviation =
          fmax(summary.maximum_pest_j_over_r2_relative_deviation, pest_value);
    }

    if (!(q_valid && flux_scale_valid && pest_valid)) {
      if (summary.first_invalid_surface < 0) {
        summary.first_invalid_surface = surface;
      }
      ++summary.invalid_surface_count;
    }
  }
  *out = summary;
}

dim3 grid_2d(int nx, int ny, dim3 block) {
  return dim3((static_cast<unsigned>(nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(ny) + block.y - 1) / block.y);
}

}  // namespace

void ToroidalEquilibriumFields::resize(int n_lambda_in, int n_theta_in) {
  if (n_lambda_in < 0 || n_theta_in < 0) {
    throw std::invalid_argument{
        "ToroidalEquilibriumFields dimensions must be non-negative"};
  }
  const std::size_t radial = static_cast<std::size_t>(n_lambda_in);
  const std::size_t geometry = checked_product(
      radial, static_cast<std::size_t>(n_theta_in),
      "ToroidalEquilibriumFields storage overflows size_t");

  // Complete every arithmetic check before asking the backend to allocate.
  // This makes extreme dimensions deterministic even without a HIP device.
  (void)backend::detail::checked_buffer_bytes<Real>(radial);
  (void)backend::detail::checked_buffer_bytes<Real>(geometry);

  backend::DeviceBuffer<Real> next_pressure{radial};
  backend::DeviceBuffer<Real> next_pressure_lambda{radial};
  backend::DeviceBuffer<Real> next_f{radial};
  backend::DeviceBuffer<Real> next_f_squared{radial};
  backend::DeviceBuffer<Real> next_f_lambda{radial};
  backend::DeviceBuffer<Real> next_ff_lambda{radial};
  backend::DeviceBuffer<Real> next_density{radial};

  backend::DeviceBuffer<Real> next_g_lambda_lambda{geometry};
  backend::DeviceBuffer<Real> next_g_lambda_theta{geometry};
  backend::DeviceBuffer<Real> next_g_theta_theta{geometry};
  backend::DeviceBuffer<Real> next_g_phi_phi{geometry};
  backend::DeviceBuffer<Real> next_jacobian{geometry};
  backend::DeviceBuffer<Real> next_b_theta{geometry};
  backend::DeviceBuffer<Real> next_b_phi{geometry};
  backend::DeviceBuffer<Real> next_j_theta{geometry};
  backend::DeviceBuffer<Real> next_j_phi{geometry};

  pressure = std::move(next_pressure);
  pressure_lambda = std::move(next_pressure_lambda);
  f = std::move(next_f);
  f_squared = std::move(next_f_squared);
  f_lambda = std::move(next_f_lambda);
  ff_lambda = std::move(next_ff_lambda);
  density = std::move(next_density);

  g_lambda_lambda = std::move(next_g_lambda_lambda);
  g_lambda_theta = std::move(next_g_lambda_theta);
  g_theta_theta = std::move(next_g_theta_theta);
  g_phi_phi = std::move(next_g_phi_phi);
  jacobian = std::move(next_jacobian);
  b_theta = std::move(next_b_theta);
  b_phi = std::move(next_b_phi);
  j_theta = std::move(next_j_theta);
  j_phi = std::move(next_j_phi);

  n_lambda = n_lambda_in;
  n_theta = n_theta_in;
  signed_flux_scale = Real{0};
}

void launch_build_toroidal_equilibrium(
    const FluxCoordinateGrid& coords,
    const equilibrium::ProfileCoefficients& profile, Real psi_axis,
    Real psi_boundary, Real profile_scale, Real f_vacuum,
    const DensityProfileCoefficients& density,
    ToroidalEquilibriumFields& out, stream_t stream) {
  const CoordinateShape shape = validate_coordinates(coords);
  validate_profile_metadata(profile, density, psi_axis, psi_boundary,
                            profile_scale, f_vacuum);
  validate_equilibrium_output(out, shape);

  const Real signed_flux_scale = psi_axis - psi_boundary;
  const Real delta_psi = psi_boundary - psi_axis;
  out.signed_flux_scale = signed_flux_scale;
  auto* hip_stream = static_cast<hipStream_t>(stream);

  const dim3 radial_block{kBlock1D};
  const dim3 radial_grid{
      (static_cast<unsigned>(shape.n_lambda) + kBlock1D - 1) / kBlock1D};
  hipLaunchKernelGGL(
      evaluate_profiles_kernel, radial_grid, radial_block, 0, hip_stream,
      coords.psi_n.device_ptr(), shape.n_lambda, profile, density, delta_psi,
      profile_scale, f_vacuum, out.pressure.device_ptr(),
      out.pressure_lambda.device_ptr(), out.f.device_ptr(),
      out.f_squared.device_ptr(), out.f_lambda.device_ptr(),
      out.ff_lambda.device_ptr(), out.density.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  const dim3 geometry_block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(
      build_geometry_kernel,
      grid_2d(shape.n_theta, shape.n_lambda, geometry_block), geometry_block,
      0, hip_stream, coords.r.device_ptr(), coords.dr_dpsi.device_ptr(),
      coords.dz_dpsi.device_ptr(), coords.dr_dtheta.device_ptr(),
      coords.dz_dtheta.device_ptr(), coords.q.device_ptr(),
      out.pressure_lambda.device_ptr(), out.f_lambda.device_ptr(),
      shape.n_lambda, shape.n_theta, signed_flux_scale,
      out.g_lambda_lambda.device_ptr(), out.g_lambda_theta.device_ptr(),
      out.g_theta_theta.device_ptr(), out.g_phi_phi.device_ptr(),
      out.jacobian.device_ptr(), out.b_theta.device_ptr(),
      out.b_phi.device_ptr(), out.j_theta.device_ptr(),
      out.j_phi.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  backend::DeviceBuffer<int> device_status{
      1, backend::on_device(shape.owner)};
  hipLaunchKernelGGL(
      validate_build_kernel, dim3{1}, dim3{1}, 0, hip_stream,
      coords.psi_n.device_ptr(), coords.q.device_ptr(),
      coords.valid.device_ptr(), shape.n_lambda, shape.n_theta,
      coords.r.device_ptr(), coords.dr_dpsi.device_ptr(),
      coords.dz_dpsi.device_ptr(), coords.dr_dtheta.device_ptr(),
      coords.dz_dtheta.device_ptr(), coords.jacobian.device_ptr(),
      out.pressure.device_ptr(), out.pressure_lambda.device_ptr(),
      out.f.device_ptr(), out.f_squared.device_ptr(),
      out.f_lambda.device_ptr(), out.ff_lambda.device_ptr(),
      out.density.device_ptr(), out.g_lambda_lambda.device_ptr(),
      out.g_lambda_theta.device_ptr(), out.g_theta_theta.device_ptr(),
      out.g_phi_phi.device_ptr(), out.jacobian.device_ptr(),
      out.b_theta.device_ptr(), out.b_phi.device_ptr(),
      out.j_theta.device_ptr(), out.j_phi.device_ptr(),
      device_status.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  backend::device_synchronize(stream);
  int host_status = 0;
  device_status.copy_to_host(&host_status, 1);
  switch (static_cast<BuildStatus>(host_status)) {
    case BuildStatus::success:
      return;
    case BuildStatus::invalid_lambda:
      throw ToroidalEquilibriumValidationError{
          ToroidalEquilibriumValidationStatus::invalid_lambda,
          "toroidal equilibrium requires valid finite lambda nodes in [0,1] "
          "and finite q"};
    case BuildStatus::nonpositive_f_squared:
      throw ToroidalEquilibriumValidationError{
          ToroidalEquilibriumValidationStatus::nonpositive_f_squared,
          "toroidal equilibrium requires F^2(lambda) > 0 at every spectral "
          "radial node"};
    case BuildStatus::nonpositive_density:
      throw ToroidalEquilibriumValidationError{
          ToroidalEquilibriumValidationStatus::nonpositive_density,
          "toroidal equilibrium requires rho(lambda) > 0 at every spectral "
          "radial node"};
    case BuildStatus::nonfinite_profile:
      throw ToroidalEquilibriumValidationError{
          ToroidalEquilibriumValidationStatus::nonfinite_profile,
          "toroidal equilibrium analytic profile evaluation produced a "
          "non-finite value"};
    case BuildStatus::invalid_geometry:
      throw ToroidalEquilibriumValidationError{
          ToroidalEquilibriumValidationStatus::invalid_geometry,
          "toroidal equilibrium requires finite, non-degenerate source "
          "geometry consistent with FluxCoordinateGrid::jacobian"};
  }
  throw std::runtime_error{"toroidal equilibrium returned an unknown status"};
}

void ToroidalGeometryValidation::resize(int n_surfaces_in) {
  if (n_surfaces_in < 0) {
    throw std::invalid_argument{
        "ToroidalGeometryValidation surface count must be non-negative"};
  }
  const std::size_t n = static_cast<std::size_t>(n_surfaces_in);
  q_relative_deviation = backend::DeviceBuffer<Real>{n};
  flux_scale_relative_deviation = backend::DeviceBuffer<Real>{n};
  pest_j_over_r2_relative_deviation = backend::DeviceBuffer<Real>{n};
  n_surfaces = n_surfaces_in;
}

void launch_validate_toroidal_geometry(
    const FluxCoordinateGrid& coords, const numerics::EllipticGrid& grid,
    const equilibrium::GsMagneticField& field,
    const ToroidalEquilibriumFields& equilibrium,
    ToroidalGeometryValidation& out, stream_t stream) {
  const CoordinateShape shape = validate_coordinates(coords);
  grid.validate();
  if (grid.nr < 4 || grid.nz < 4) {
    throw std::invalid_argument{
        "toroidal geometry validation requires at least four equilibrium-grid "
        "nodes per axis for cubic interpolation"};
  }
  validate_equilibrium_output(equilibrium, shape);
  if (!std::isfinite(equilibrium.signed_flux_scale)
      || equilibrium.signed_flux_scale == Real{0}) {
    throw std::invalid_argument{
        "toroidal geometry validation requires a finite nonzero signed flux "
        "scale"};
  }

  require_size(field.b_r, grid.size(), "GsMagneticField::b_r");
  require_size(field.b_z, grid.size(), "GsMagneticField::b_z");
  require_size(field.b_phi, grid.size(), "GsMagneticField::b_phi");
  require_owner(field.b_r, shape.owner, "GsMagneticField::b_r");
  require_owner(field.b_z, shape.owner, "GsMagneticField::b_z");
  require_owner(field.b_phi, shape.owner, "GsMagneticField::b_phi");

  if (out.n_surfaces != shape.n_lambda) {
    throw std::invalid_argument{
        "ToroidalGeometryValidation shape must match FluxCoordinateGrid"};
  }
  require_size(out.q_relative_deviation, shape.radial_points,
               "ToroidalGeometryValidation::q_relative_deviation");
  require_size(out.flux_scale_relative_deviation, shape.radial_points,
               "ToroidalGeometryValidation::flux_scale_relative_deviation");
  require_size(
      out.pest_j_over_r2_relative_deviation, shape.radial_points,
      "ToroidalGeometryValidation::pest_j_over_r2_relative_deviation");
  require_owner(out.q_relative_deviation, shape.owner,
                "ToroidalGeometryValidation::q_relative_deviation");
  require_owner(out.flux_scale_relative_deviation, shape.owner,
                "ToroidalGeometryValidation::flux_scale_relative_deviation");
  require_owner(
      out.pest_j_over_r2_relative_deviation, shape.owner,
      "ToroidalGeometryValidation::pest_j_over_r2_relative_deviation");

  const dim3 block{kBlock1D};
  const dim3 launch_grid{
      (static_cast<unsigned>(shape.n_lambda) + kBlock1D - 1) / kBlock1D};
  hipLaunchKernelGGL(
      validate_geometry_kernel, launch_grid, block, 0,
      static_cast<hipStream_t>(stream), view_of(grid),
      coords.r.device_ptr(), coords.z.device_ptr(),
      coords.dr_dpsi.device_ptr(), coords.dz_dpsi.device_ptr(),
      coords.dr_dtheta.device_ptr(), coords.dz_dtheta.device_ptr(),
      coords.q.device_ptr(), coords.valid.device_ptr(), shape.n_lambda,
      shape.n_theta, field.b_r.device_ptr(), field.b_z.device_ptr(),
      field.b_phi.device_ptr(), equilibrium.signed_flux_scale,
      out.q_relative_deviation.device_ptr(),
      out.flux_scale_relative_deviation.device_ptr(),
      out.pest_j_over_r2_relative_deviation.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());
}

ToroidalGeometryValidationSummary summarize_toroidal_geometry_validation(
    const ToroidalGeometryValidation& validation, stream_t stream) {
  if (validation.n_surfaces <= 0) {
    throw std::invalid_argument{
        "toroidal geometry summary requires at least one surface"};
  }
  const std::size_t n = static_cast<std::size_t>(validation.n_surfaces);
  require_size(validation.q_relative_deviation, n,
               "ToroidalGeometryValidation::q_relative_deviation");
  require_size(validation.flux_scale_relative_deviation, n,
               "ToroidalGeometryValidation::flux_scale_relative_deviation");
  require_size(
      validation.pest_j_over_r2_relative_deviation, n,
      "ToroidalGeometryValidation::pest_j_over_r2_relative_deviation");

  const int owner = validation.q_relative_deviation.owner_device();
  if (owner < 0) {
    throw std::invalid_argument{
        "toroidal geometry validation is not device-resident"};
  }
  require_owner(validation.flux_scale_relative_deviation, owner,
                "ToroidalGeometryValidation::flux_scale_relative_deviation");
  require_owner(
      validation.pest_j_over_r2_relative_deviation, owner,
      "ToroidalGeometryValidation::pest_j_over_r2_relative_deviation");
  if (backend::current_device() != owner) {
    throw std::invalid_argument{
        "toroidal geometry validation is not on the current device"};
  }

  backend::DeviceGuard guard{owner};
  backend::DeviceBuffer<ToroidalGeometryValidationSummary> device_summary{
      1, backend::uninitialized, backend::on_device(owner)};
  hipLaunchKernelGGL(
      summarize_geometry_validation_kernel, dim3{1}, dim3{1}, 0,
      static_cast<hipStream_t>(stream),
      validation.q_relative_deviation.device_ptr(),
      validation.flux_scale_relative_deviation.device_ptr(),
      validation.pest_j_over_r2_relative_deviation.device_ptr(),
      validation.n_surfaces, device_summary.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  ToroidalGeometryValidationSummary summary{};
  device_summary.copy_to_host_async(&summary, 1, stream);
  backend::device_synchronize(stream);
  return summary;
}

}  // namespace quasar::stability
