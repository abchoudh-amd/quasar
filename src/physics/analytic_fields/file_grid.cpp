#include "quasar/physics/analytic_fields/file_grid.hpp"

#include "quasar/core/observations.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace quasar::analytic_fields {

namespace {

bool finite(Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y)
      && std::isfinite(value.z);
}

std::size_t checked_product(std::array<std::size_t, 3> dims) {
  std::size_t product = 1;
  for (const std::size_t n : dims) {
    if (n == 0) {
      throw std::invalid_argument{"FileGridEvaluator: dimensions must be positive"};
    }
    if (product > std::numeric_limits<std::size_t>::max() / n) {
      throw std::overflow_error{"FileGridEvaluator: grid size is not representable"};
    }
    product *= n;
  }
  return product;
}

struct ScaledTerm {
  long double mantissa{0.0L};
  int exponent{0};
};

ScaledTerm scaled_factor(long double value) {
  if (value == 0.0L) return {};
  if (!std::isfinite(value)) {
    throw std::overflow_error{
        "FileGridEvaluator: intermediate factor is not representable"};
  }
  int exponent = 0;
  const long double mantissa = std::frexp(value, &exponent);
  return ScaledTerm{mantissa, exponent};
}

ScaledTerm scaled_product(std::initializer_list<ScaledTerm> factors) {
  ScaledTerm product{1.0L, 0};
  for (const ScaledTerm factor : factors) {
    if (factor.mantissa == 0.0L) return {};
    product.mantissa *= factor.mantissa;
    product.exponent += factor.exponent;
    int adjustment = 0;
    product.mantissa = std::frexp(product.mantissa, &adjustment);
    product.exponent += adjustment;
  }
  return product;
}

template <std::size_t Capacity>
ScaledTerm scaled_sum(std::array<ScaledTerm, Capacity> terms,
                      std::size_t count, bool absolute = false) {
  std::array<ScaledTerm, Capacity> active{};
  std::size_t active_count = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (terms[i].mantissa == 0.0L) continue;
    if (absolute) terms[i].mantissa = std::abs(terms[i].mantissa);
    active[active_count++] = terms[i];
  }

  // Combine equal-scale terms before smaller ones.  This preserves exact
  // cancellation of large terms without first discarding a small residual,
  // and never forms the potentially overflowing unscaled values.
  while (active_count > 1) {
    std::size_t first = 0;
    for (std::size_t i = 1; i < active_count; ++i) {
      if (active[i].exponent > active[first].exponent) first = i;
    }
    std::size_t second = first == 0 ? 1 : 0;
    for (std::size_t i = 0; i < active_count; ++i) {
      if (i != first && active[i].exponent > active[second].exponent) {
        second = i;
      }
    }
    const ScaledTerm a = active[first];
    const ScaledTerm b = active[second];
    const long double sum = a.mantissa
        + std::scalbn(b.mantissa, b.exponent - a.exponent);

    std::array<ScaledTerm, Capacity> remaining{};
    std::size_t remaining_count = 0;
    for (std::size_t i = 0; i < active_count; ++i) {
      if (i != first && i != second) remaining[remaining_count++] = active[i];
    }
    if (sum != 0.0L) {
      int adjustment = 0;
      const long double mantissa = std::frexp(sum, &adjustment);
      remaining[remaining_count++] =
          ScaledTerm{mantissa, a.exponent + adjustment};
    }
    active = remaining;
    active_count = remaining_count;
  }
  return active_count == 0 ? ScaledTerm{} : active[0];
}

Real checked_real(ScaledTerm value, const char* what) {
  if (value.mantissa == 0.0L) return Real{0};
  int max_exponent = 0;
  const long double max_mantissa = std::frexp(
      static_cast<long double>(std::numeric_limits<Real>::max()),
      &max_exponent);
  const long double magnitude = std::abs(value.mantissa);
  if (value.exponent > max_exponent
      || (value.exponent == max_exponent && magnitude > max_mantissa)) {
    throw std::overflow_error{std::string{"FileGridEvaluator: "} + what
                              + " is not representable"};
  }
  const Real result = std::scalbn(static_cast<Real>(value.mantissa),
                                  value.exponent);
  if (!std::isfinite(result)) {
    throw std::overflow_error{std::string{"FileGridEvaluator: "} + what
                              + " is not representable"};
  }
  return result;
}

ScaledTerm reciprocal_factor(Real value, long double sign) {
  int exponent = 0;
  const Real mantissa = std::frexp(value, &exponent);
  return scaled_product(
      {scaled_factor(sign / static_cast<long double>(mantissa)),
       ScaledTerm{0.5L, 1 - exponent}});
}

bool magnitude_greater(ScaledTerm lhs, ScaledTerm rhs) noexcept {
  if (lhs.mantissa == 0.0L) return false;
  if (rhs.mantissa == 0.0L) return true;
  if (lhs.exponent != rhs.exponent) return lhs.exponent > rhs.exponent;
  return std::abs(lhs.mantissa) > std::abs(rhs.mantissa);
}

ScaledTerm difference_quotient(Real high, Real low, Real denominator) {
  const std::array<ScaledTerm, 2> difference_terms{
      scaled_factor(static_cast<long double>(high)),
      scaled_factor(-static_cast<long double>(low))};
  return scaled_product(
      {scaled_sum(difference_terms, difference_terms.size()),
       reciprocal_factor(denominator, 1.0L)});
}

Real grid_coordinate(Real origin, Real spacing, std::size_t index) {
  const Real coordinate = std::fma(static_cast<Real>(index), spacing, origin);
  if (!std::isfinite(coordinate)) {
    throw std::overflow_error{
        "FileGridEvaluator: upper grid coordinate is not representable"};
  }
  return coordinate;
}

bool dimension_fits_size_t(Real value) {
  constexpr int size_bits = std::numeric_limits<std::size_t>::digits;
  constexpr int real_max_exponent = std::numeric_limits<Real>::max_exponent;
  if constexpr (size_bits < real_max_exponent) {
    // size_t is unsigned, hence its first unrepresentable positive integer is
    // exactly 2^digits.  Comparing against that exclusive bound avoids casting
    // SIZE_MAX to a floating type that may round it up to 2^digits.
    return value < std::ldexp(Real{1}, size_bits);
  }
  // Every finite non-negative Real then lies below size_t's upper bound.
  return true;
}

struct AxisWeights {
  std::size_t base{0};
  int count{1};
  std::array<long double, 2> weight{1.0L, 0.0L};
  std::array<ScaledTerm, 2> derivative{};
};

AxisWeights weights_for_axis(Real coordinate, Real origin, Real spacing,
                             std::size_t n, const char* axis) {
  if (!std::isfinite(coordinate)) {
    throw std::invalid_argument{std::string{"FileGridEvaluator: observation "}
                                + axis + " coordinate must be finite"};
  }
  if (n == 1) {
    // A singleton axis denotes one geometric plane, not an invariant
    // direction.  Accepting nearby-but-distinct floating-point values silently
    // projects observations onto the wrong plane, especially at large origins.
    if (coordinate != origin) {
      throw std::out_of_range{std::string{"FileGridEvaluator: observation "}
                              + axis + " coordinate lies outside the field grid"};
    }
    return {};
  }
  const ScaledTerm scaled_u = difference_quotient(coordinate, origin, spacing);
  const long double u = std::scalbn(scaled_u.mantissa, scaled_u.exponent);
  const long double last = static_cast<long double>(n - 1);
  const long double tolerance = 64.0L
      * static_cast<long double>(std::numeric_limits<Real>::epsilon())
      * std::max(1.0L, last);
  if (!std::isfinite(u) || u < -tolerance || u > last + tolerance) {
    throw std::out_of_range{std::string{"FileGridEvaluator: observation "}
                            + axis + " coordinate lies outside the field grid"};
  }
  const long double clamped = std::clamp(u, 0.0L, last);
  const std::size_t base = clamped >= last
      ? n - 2
      : static_cast<std::size_t>(std::floor(clamped));
  const long double fraction = clamped - static_cast<long double>(base);
  AxisWeights out;
  out.base = base;
  out.count = 2;
  out.weight = {1.0L - fraction, fraction};
  out.derivative = {reciprocal_factor(spacing, -1.0L),
                    reciprocal_factor(spacing, 1.0L)};
  return out;
}

std::size_t flat_index(const std::array<std::size_t, 3>& dims,
                       std::size_t i, std::size_t j, std::size_t k) {
  return i + dims[0] * (j + dims[1] * k);
}

template <class Grid, class WeightX, class WeightY, class WeightZ>
Vec3 weighted_sum(const Grid& grid,
                  const AxisWeights& x, const AxisWeights& y,
                  const AxisWeights& z, WeightX wx, WeightY wy, WeightZ wz,
                  const char* what) {
  std::array<std::array<ScaledTerm, 8>, 3> terms{};
  std::size_t count = 0;
  for (int ck = 0; ck < z.count; ++ck) {
    for (int cj = 0; cj < y.count; ++cj) {
      for (int ci = 0; ci < x.count; ++ci) {
        const ScaledTerm weight =
            scaled_product({wx(x, ci), wy(y, cj), wz(z, ck)});
        const Vec3 value = grid.values[flat_index(
            grid.dims, x.base + static_cast<std::size_t>(ci),
            y.base + static_cast<std::size_t>(cj),
            z.base + static_cast<std::size_t>(ck))];
        terms[0][count] = scaled_product(
            {weight, scaled_factor(static_cast<long double>(value.x))});
        terms[1][count] = scaled_product(
            {weight, scaled_factor(static_cast<long double>(value.y))});
        terms[2][count] = scaled_product(
            {weight, scaled_factor(static_cast<long double>(value.z))});
        ++count;
      }
    }
  }
  return Vec3{checked_real(scaled_sum(terms[0], count), what),
              checked_real(scaled_sum(terms[1], count), what),
              checked_real(scaled_sum(terms[2], count), what)};
}

}  // namespace

FileGridEvaluator::FileGridEvaluator(std::string path) : path_{std::move(path)} {
  set_grid(load_text_grid(path_));
}

void FileGridEvaluator::validate_grid(const GridData& grid) {
  if (!finite(grid.origin) || !finite(grid.spacing)
      || !std::isfinite(grid.divergence_tolerance)
      || grid.divergence_tolerance < Real{0}) {
    throw std::invalid_argument{
        "FileGridEvaluator: origin, spacing, and divergence tolerance must be finite"};
  }
  const std::size_t expected = checked_product(grid.dims);
  if (grid.values.size() != expected) {
    throw std::invalid_argument{
        "FileGridEvaluator: values length does not match nx*ny*nz"};
  }
  for (const Vec3 value : grid.values) {
    if (!finite(value)) {
      throw std::invalid_argument{
          "FileGridEvaluator: field values must have finite components"};
    }
  }
  const Real origins[] = {grid.origin.x, grid.origin.y, grid.origin.z};
  const Real spacings[] = {grid.spacing.x, grid.spacing.y, grid.spacing.z};
  for (int axis = 0; axis < 3; ++axis) {
    if (spacings[axis] < Real{0}
        || (grid.dims[axis] > 1 && spacings[axis] <= Real{0})) {
      throw std::invalid_argument{
          "FileGridEvaluator: spacing must be positive on non-singleton axes"};
    }
    if (grid.dims[axis] > 1) {
      const Real first_next = grid_coordinate(
          origins[axis], spacings[axis], std::size_t{1});
      const Real upper = grid_coordinate(
          origins[axis], spacings[axis], grid.dims[axis] - 1);
      const Real upper_previous = grid_coordinate(
          origins[axis], spacings[axis], grid.dims[axis] - 2);
      if (first_next == origins[axis] || upper == upper_previous) {
        throw std::overflow_error{
            "FileGridEvaluator: adjacent grid coordinates collapse in host precision"};
      }
    }
  }

  // Componentwise trilinear interpolation does not automatically preserve
  // div(B)=0. Its divergence is multi-affine inside each cell, so its extrema
  // occur at cell vertices. Check those vertices and reject a map whose local
  // monopole error exceeds the configured relative tolerance (plus a round-off
  // allowance). This keeps mildly discretized solenoidal maps usable while
  // preventing a grossly non-physical field from being silently accepted.
  const std::size_t cells_x = grid.dims[0] > 1 ? grid.dims[0] - 1 : 1;
  const std::size_t cells_y = grid.dims[1] > 1 ? grid.dims[1] - 1 : 1;
  const std::size_t cells_z = grid.dims[2] > 1 ? grid.dims[2] - 1 : 1;
  const int vertices_x = grid.dims[0] > 1 ? 2 : 1;
  const int vertices_y = grid.dims[1] > 1 ? 2 : 1;
  const int vertices_z = grid.dims[2] > 1 ? 2 : 1;
  const long double relative_tolerance =
      static_cast<long double>(grid.divergence_tolerance)
      + 256.0L * static_cast<long double>(std::numeric_limits<Real>::epsilon());
  for (std::size_t k = 0; k < cells_z; ++k) {
    for (std::size_t j = 0; j < cells_y; ++j) {
      for (std::size_t i = 0; i < cells_x; ++i) {
        for (int vk = 0; vk < vertices_z; ++vk) {
          for (int vj = 0; vj < vertices_y; ++vj) {
            for (int vi = 0; vi < vertices_x; ++vi) {
              std::array<ScaledTerm, 3> derivatives{};
              if (grid.dims[0] > 1) {
                const Vec3 lo = grid.values[flat_index(
                    grid.dims, i, j + static_cast<std::size_t>(vj),
                    k + static_cast<std::size_t>(vk))];
                const Vec3 hi = grid.values[flat_index(
                    grid.dims, i + 1, j + static_cast<std::size_t>(vj),
                    k + static_cast<std::size_t>(vk))];
                derivatives[0] =
                    difference_quotient(hi.x, lo.x, grid.spacing.x);
              }
              if (grid.dims[1] > 1) {
                const Vec3 lo = grid.values[flat_index(
                    grid.dims, i + static_cast<std::size_t>(vi), j,
                    k + static_cast<std::size_t>(vk))];
                const Vec3 hi = grid.values[flat_index(
                    grid.dims, i + static_cast<std::size_t>(vi), j + 1,
                    k + static_cast<std::size_t>(vk))];
                derivatives[1] =
                    difference_quotient(hi.y, lo.y, grid.spacing.y);
              }
              if (grid.dims[2] > 1) {
                const Vec3 lo = grid.values[flat_index(
                    grid.dims, i + static_cast<std::size_t>(vi),
                    j + static_cast<std::size_t>(vj), k)];
                const Vec3 hi = grid.values[flat_index(
                    grid.dims, i + static_cast<std::size_t>(vi),
                    j + static_cast<std::size_t>(vj), k + 1)];
                derivatives[2] =
                    difference_quotient(hi.z, lo.z, grid.spacing.z);
              }
              const ScaledTerm divergence =
                  scaled_sum(derivatives, derivatives.size());
              const ScaledTerm scale =
                  scaled_sum(derivatives, derivatives.size(), true);
              const ScaledTerm threshold = scaled_product(
                  {scale, scaled_factor(relative_tolerance)});
              if (magnitude_greater(divergence, threshold)) {
                throw std::invalid_argument{
                    "FileGridEvaluator: trilinear field map is not solenoidal"};
              }
            }
          }
        }
      }
    }
  }
}

void FileGridEvaluator::set_grid(GridData grid) {
  validate_grid(grid);
  grid_ = std::move(grid);
  configured_ = true;
}

FileGridEvaluator::GridData FileGridEvaluator::load_text_grid(
    const std::string& path) {
  if (path.empty()) {
    throw std::invalid_argument{"FileGridEvaluator: path must not be empty"};
  }
  std::ifstream input{path};
  if (!input) {
    throw std::runtime_error{"FileGridEvaluator: cannot open field grid '" + path + "'"};
  }

  std::string magic;
  int version = 0;
  if (!(input >> magic >> version) || magic != "QUASAR_FILE_GRID" || version != 1) {
    throw std::invalid_argument{
        "FileGridEvaluator: expected 'QUASAR_FILE_GRID 1' header"};
  }

  GridData grid;
  std::string keyword;
  long long nx = 0, ny = 0, nz = 0;
  if (!(input >> keyword >> nx >> ny >> nz) || keyword != "dims") {
    throw std::invalid_argument{"FileGridEvaluator: expected 'dims nx ny nz'"};
  }
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    throw std::invalid_argument{"FileGridEvaluator: dimensions must be positive"};
  }
  const auto fits_size_t = [](long long value) {
    if constexpr (sizeof(std::size_t) < sizeof(long long)) {
      return static_cast<unsigned long long>(value)
          <= static_cast<unsigned long long>(
              std::numeric_limits<std::size_t>::max());
    }
    return true;
  };
  if (!fits_size_t(nx) || !fits_size_t(ny) || !fits_size_t(nz)) {
    throw std::overflow_error{
        "FileGridEvaluator: dimension is not representable as size_t"};
  }
  grid.dims = {static_cast<std::size_t>(nx), static_cast<std::size_t>(ny),
               static_cast<std::size_t>(nz)};
  if (!(input >> keyword >> grid.origin.x >> grid.origin.y >> grid.origin.z)
      || keyword != "origin") {
    throw std::invalid_argument{"FileGridEvaluator: expected 'origin x y z'"};
  }
  if (!(input >> keyword >> grid.spacing.x >> grid.spacing.y >> grid.spacing.z)
      || keyword != "spacing") {
    throw std::invalid_argument{"FileGridEvaluator: expected 'spacing dx dy dz'"};
  }
  if (!(input >> keyword) || keyword != "data") {
    throw std::invalid_argument{"FileGridEvaluator: expected 'data' marker"};
  }

  const std::size_t count = checked_product(grid.dims);
  grid.values.resize(count);
  for (std::size_t index = 0; index < count; ++index) {
    Vec3& value = grid.values[index];
    if (!(input >> value.x >> value.y >> value.z)) {
      throw std::invalid_argument{
          "FileGridEvaluator: field grid ended before nx*ny*nz vectors"};
    }
  }
  std::string extra;
  if (input >> extra) {
    throw std::invalid_argument{
        "FileGridEvaluator: unexpected trailing data after field vectors"};
  }
  validate_grid(grid);
  return grid;
}

void FileGridEvaluator::configure(const numerics::EvaluatorParams& params) {
  numerics::reject_unknown_params(
      params, {"origin", "spacing", "dims", "values"},
      "FileGridEvaluator");
  const auto origin_it = params.find("origin");
  const auto spacing_it = params.find("spacing");
  const auto dims_it = params.find("dims");
  const auto values_it = params.find("values");
  if (origin_it == params.end() || spacing_it == params.end()
      || dims_it == params.end() || values_it == params.end()) {
    throw std::invalid_argument{
        "FileGridEvaluator: configure requires origin, spacing, dims, and values"};
  }
  GridData grid;
  grid.origin = numerics::param_vec3(params, "origin");
  grid.spacing = numerics::param_vec3(params, "spacing");
  if (dims_it->second.size() != 3) {
    throw std::invalid_argument{"FileGridEvaluator: dims must have 3 elements"};
  }
  for (int axis = 0; axis < 3; ++axis) {
    const Real value = dims_it->second[axis];
    if (!std::isfinite(value) || value < Real{1}
        || std::floor(value) != value || !dimension_fits_size_t(value)) {
      throw std::invalid_argument{
          "FileGridEvaluator: dims must be positive exact integers"};
    }
    grid.dims[axis] = static_cast<std::size_t>(value);
  }
  const std::size_t count = checked_product(grid.dims);
  if (count > std::numeric_limits<std::size_t>::max() / 3) {
    throw std::overflow_error{
        "FileGridEvaluator: flattened vector count is not representable"};
  }
  if (values_it->second.size() != 3 * count) {
    throw std::invalid_argument{
        "FileGridEvaluator: values must contain 3*nx*ny*nz elements"};
  }
  grid.values.resize(count);
  for (std::size_t index = 0; index < count; ++index) {
    grid.values[index] = Vec3{values_it->second[3 * index],
                              values_it->second[3 * index + 1],
                              values_it->second[3 * index + 2]};
  }
  set_grid(std::move(grid));
  path_.clear();
}

Field<Vec3> FileGridEvaluator::evaluate_B(const core::IFieldSource&,
                                          const core::PointCloud& observations) const {
  if (!configured_) {
    throw std::invalid_argument{"FileGridEvaluator: no field grid is configured"};
  }
  Field<Vec3> out(observations.size());
  const auto regular = [](const AxisWeights& a, int i) {
    return scaled_factor(a.weight[i]);
  };
  const auto& points = observations.points();
  for (std::size_t index = 0; index < points.size(); ++index) {
    const AxisWeights x = weights_for_axis(
        points[index].x, grid_.origin.x, grid_.spacing.x, grid_.dims[0], "x");
    const AxisWeights y = weights_for_axis(
        points[index].y, grid_.origin.y, grid_.spacing.y, grid_.dims[1], "y");
    const AxisWeights z = weights_for_axis(
        points[index].z, grid_.origin.z, grid_.spacing.z, grid_.dims[2], "z");
    out[index] = weighted_sum(grid_, x, y, z, regular, regular, regular,
                              "interpolated magnetic field");
  }
  return out;
}

Field<Mat3x3> FileGridEvaluator::evaluate_grad_B(const core::IFieldSource&,
                                                 const core::PointCloud& observations) const {
  if (!configured_) {
    throw std::invalid_argument{"FileGridEvaluator: no field grid is configured"};
  }
  Field<Mat3x3> out(observations.size());
  const auto regular = [](const AxisWeights& a, int i) {
    return scaled_factor(a.weight[i]);
  };
  const auto derivative = [](const AxisWeights& a, int i) {
    return a.derivative[i];
  };
  const auto& points = observations.points();
  for (std::size_t index = 0; index < points.size(); ++index) {
    const AxisWeights x = weights_for_axis(
        points[index].x, grid_.origin.x, grid_.spacing.x, grid_.dims[0], "x");
    const AxisWeights y = weights_for_axis(
        points[index].y, grid_.origin.y, grid_.spacing.y, grid_.dims[1], "y");
    const AxisWeights z = weights_for_axis(
        points[index].z, grid_.origin.z, grid_.spacing.z, grid_.dims[2], "z");
    const Vec3 d_dx = weighted_sum(grid_, x, y, z, derivative, regular, regular,
                                   "magnetic-field x derivative");
    const Vec3 d_dy = weighted_sum(grid_, x, y, z, regular, derivative, regular,
                                   "magnetic-field y derivative");
    const Vec3 d_dz = weighted_sum(grid_, x, y, z, regular, regular, derivative,
                                   "magnetic-field z derivative");
    // Mat3x3 stores rows (field component) while each weighted derivative above
    // is a column dB/dcoordinate.
    out[index] = Mat3x3{Vec3{d_dx.x, d_dy.x, d_dz.x},
                        Vec3{d_dx.y, d_dy.y, d_dz.y},
                        Vec3{d_dx.z, d_dy.z, d_dz.z}};
  }
  return out;
}

QUASAR_REGISTER_FIELD_EVALUATOR("file_grid", FileGridEvaluator)

}  // namespace quasar::analytic_fields
