#include "quasar/numerics/radial_tables.hpp"

#include "quasar/numerics/finite_volume_quadrature.hpp"
#include "quasar/numerics/radial_moments.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace quasar::numerics {
namespace {

constexpr Real kMaximumAcceptedResidual = Real{1e-11};

int collocation_width(int scheme_order) {
  return scheme_order == 7 ? 8 : scheme_order == 5 ? 6 : 4;
}

Real reduced_cell_radius(const Grid2D& grid, int cell) {
  // Do not replace this accessor with cell+1/2.  On a distributed tile it uses
  // Grid2D's canonical global-x mapping, avoiding the nested-FMA rounding of a
  // tile-local origin and making overlapping rows bit-identical.
  return grid.r_at_cell_center(cell) / grid.dx();
}

bool is_axis_face(const Grid2D& grid, int face) {
  return grid.r_at_edge(face) == Real{0};
}

// Two-pass collector for the stencil rows.
//
// The construction loop below emits about twenty independent small systems per
// radial index. `record` mode stashes each specification and returns a
// placeholder that satisfies the same validation the real row will face, so the
// loop body can be written once; `solve()` then runs the whole set through one
// batched factorization, and a second traversal of the identical loop replays
// the results in order.
//
// The cursor discipline is the invariant that makes this safe: the two
// traversals must request rows in exactly the same order, and `replay` throws
// rather than returning a mismatched row if they ever diverge.
class BatchedRowBuilder {
 public:
  RadialStencilRow row(const RadialRowSpec& spec) {
    if (!solved_) {
      row_specs_.push_back(spec);
      return placeholder(spec.width);
    }
    return replay(rows_, row_cursor_, spec.width, "stencil");
  }

  Real extrapolation(Real rho_center, int neighbor_offset, Real face_offset,
                     RadialCellMeasure measure) {
    if (!solved_) {
      extrapolation_specs_.push_back(RadialExtrapolationSpec{
          rho_center, face_offset, neighbor_offset, measure});
      return Real{0};
    }
    if (extrapolation_cursor_ >= extrapolations_.size()) {
      throw std::runtime_error{
          "RadialTables: extrapolation request order diverged between passes"};
    }
    return extrapolations_[extrapolation_cursor_++];
  }

  RadialStencilRow gauss(Real rho_anchor, int count, const Real* node_xi,
                         const Real* cartesian_weights,
                         RadialCellMeasure measure) {
    if (!solved_) {
      RadialGaussSpec spec{};
      spec.rho_anchor = rho_anchor;
      spec.count = count;
      spec.measure = measure;
      for (int q = 0; q < count; ++q) {
        spec.node_xi[q] = node_xi[q];
        spec.cartesian_weights[q] = cartesian_weights[q];
      }
      gauss_specs_.push_back(spec);
      return placeholder(count);
    }
    return replay(gauss_rows_, gauss_cursor_, count, "Gauss");
  }

  void solve() {
    rows_ = solve_radial_rows(row_specs_);
    gauss_rows_ = radial_gauss_weight_rows(gauss_specs_);
    extrapolations_ = radial_face_extrapolation_factors(extrapolation_specs_);
    solved_ = true;
  }

 private:
  // Width-correct, finite, exactly partitioned: passes validate_row so the
  // collect pass cannot trip a check that the real data would not.
  static RadialStencilRow placeholder(int width) {
    RadialStencilRow row{};
    row.width = width;
    row.c[0] = Real{1};
    return row;
  }

  static RadialStencilRow replay(const std::vector<RadialStencilRow>& source,
                                 std::size_t& cursor, int width,
                                 const char* what) {
    if (cursor >= source.size() || source[cursor].width != width) {
      throw std::runtime_error{
          std::string{"RadialTables: "} + what
          + " row request order diverged between passes"};
    }
    return source[cursor++];
  }

  std::vector<RadialRowSpec> row_specs_{};
  std::vector<RadialGaussSpec> gauss_specs_{};
  std::vector<RadialExtrapolationSpec> extrapolation_specs_{};
  std::vector<RadialStencilRow> rows_{};
  std::vector<RadialStencilRow> gauss_rows_{};
  std::vector<Real> extrapolations_{};
  std::size_t row_cursor_{0};
  std::size_t gauss_cursor_{0};
  std::size_t extrapolation_cursor_{0};
  bool solved_{false};
};

void validate_row(const RadialStencilRow& row, int expected_width,
                  const char* family, int logical_index,
                  bool residual_is_diagnostic_only = false) {
  if (row.width != expected_width) {
    throw std::runtime_error{std::string{"RadialTables: unexpected width in "} +
                             family + " row " +
                             std::to_string(logical_index)};
  }
  if (!std::isfinite(row.residual) ||
      (!residual_is_diagnostic_only &&
       row.residual > kMaximumAcceptedResidual)) {
    throw std::runtime_error{std::string{"RadialTables: ill-conditioned "} +
                             family + " row " +
                             std::to_string(logical_index)};
  }
  for (int k = 0; k < expected_width; ++k) {
    if (!std::isfinite(row.c[static_cast<std::size_t>(k)])) {
      throw std::runtime_error{std::string{"RadialTables: non-finite "} +
                             family + " coefficient in row " +
                             std::to_string(logical_index)};
    }
  }
}

void append_checked_row(std::vector<Real>& destination,
                        const RadialStencilRow& row, int expected_width,
                        const char* family, int logical_index,
                        bool residual_is_diagnostic_only = false) {
  validate_row(row, expected_width, family, logical_index,
               residual_is_diagnostic_only);
  for (int k = 0; k < expected_width; ++k) {
    destination.push_back(row.c[static_cast<std::size_t>(k)]);
  }
}

void copy_to_device(backend::DeviceBuffer<Real>& destination,
                    const std::vector<Real>& source) {
  if (source.empty()) return;
  destination = backend::DeviceBuffer<Real>{source.size(), backend::uninitialized};
  destination.copy_from_host(source.data(), source.size());
}

}  // namespace

RadialTables::RadialTables(const Grid2D& grid, int scheme_order)
    : active_{1},
      scheme_order_{scheme_order},
      radial_lo_{-grid.nghost},
      radial_count_{grid.pitch()},
      reconstruction_width_{scheme_order == 5 ? 5 : scheme_order == 7 ? 7 : 0},
      quadrature_nodes_{scheme_order == 5 ? 3 : scheme_order == 7 ? 4 : 0},
      collocation_width_{collocation_width(scheme_order)} {
  if (scheme_order != 2 && scheme_order != 5 && scheme_order != 7) {
    throw std::invalid_argument{
        "RadialTables: scheme_order must be 2, 5, or 7"};
  }
  if (grid.pitch() < collocation_width_) {
    throw std::invalid_argument{
        "RadialTables: the padded radial grid is smaller than the "
        "scheme's collocation stencil"};
  }

  const std::size_t rows = static_cast<std::size_t>(radial_count_);
  auto& r1_left = r1_left_host_;
  auto& r1_right = r1_right_host_;
  auto& r1_mphi_left = r1_mphi_left_host_;
  auto& r1_mphi_right = r1_mphi_right_host_;
  auto& r1_bphi_left = r1_bphi_left_host_;
  auto& r1_bphi_right = r1_bphi_right_host_;
  auto& r2_points = r2_points_host_;
  auto& r2_mphi_points = r2_mphi_points_host_;
  auto& r2_bphi_points = r2_bphi_points_host_;
  auto& r3_weights = r3_weights_host_;
  auto& r3_mphi_weights = r3_mphi_weights_host_;
  auto& r3_bphi_weights = r3_bphi_weights_host_;
  auto& r4_face_to_cell = r4_face_to_cell_host_;
  auto& r5_cell_to_face = r5_cell_to_face_host_;
  auto& r5_bphi_cell_to_face = r5_bphi_cell_to_face_host_;
  auto& r6_limiter = r6_limiter_host_;
  auto& r6_mphi_limiter = r6_mphi_limiter_host_;
  auto& r6_bphi_limiter = r6_bphi_limiter_host_;
  r1_left.reserve(rows * static_cast<std::size_t>(reconstruction_width_));
  r1_right.reserve(rows * static_cast<std::size_t>(reconstruction_width_));
  r1_mphi_left.reserve(
      rows * static_cast<std::size_t>(reconstruction_width_));
  r1_mphi_right.reserve(
      rows * static_cast<std::size_t>(reconstruction_width_));
  r1_bphi_left.reserve(
      rows * static_cast<std::size_t>(reconstruction_width_));
  r1_bphi_right.reserve(
      rows * static_cast<std::size_t>(reconstruction_width_));
  r2_points.reserve(rows * static_cast<std::size_t>(reconstruction_width_) *
                    static_cast<std::size_t>(quadrature_nodes_));
  r2_mphi_points.reserve(
      rows * static_cast<std::size_t>(reconstruction_width_) *
      static_cast<std::size_t>(quadrature_nodes_));
  r2_bphi_points.reserve(
      rows * static_cast<std::size_t>(reconstruction_width_) *
      static_cast<std::size_t>(quadrature_nodes_));
  r3_weights.reserve(rows * static_cast<std::size_t>(quadrature_nodes_));
  r3_mphi_weights.reserve(
      rows * static_cast<std::size_t>(quadrature_nodes_));
  r3_bphi_weights.reserve(
      rows * static_cast<std::size_t>(quadrature_nodes_));
  r4_face_to_cell.reserve(rows * static_cast<std::size_t>(collocation_width_));
  r5_cell_to_face.reserve(rows * static_cast<std::size_t>(collocation_width_));
  r5_bphi_cell_to_face.reserve(
      rows * static_cast<std::size_t>(collocation_width_));
  r6_limiter.reserve(rows * std::size_t{4});
  r6_mphi_limiter.reserve(rows * std::size_t{4});
  r6_bphi_limiter.reserve(rows * std::size_t{4});

  // Every stencil row below is an independent small system, and there are about
  // twenty of them per radial index. Solving them one at a time would spend all
  // its time on launch overhead, so the loop runs twice: once to collect the
  // specifications, then once more to consume the batched results in the same
  // order. In the collect pass `solve_row`/`gauss_row` record their argument and
  // hand back a harmless placeholder, so the loop body itself is written once.
  BatchedRowBuilder batch;
  const auto solve_row = [&batch](Real rho_anchor, int width, int offset,
                                  RadialMomentTarget target,
                                  RadialCellMeasure measure, Real node_xi) {
    return batch.row(
        RadialRowSpec{rho_anchor, node_xi, width, offset, target, measure});
  };
  const auto gauss_row = [&batch](Real rho_anchor, int count,
                                  const Real* node_xi,
                                  const Real* cartesian_weights,
                                  RadialCellMeasure measure) {
    return batch.gauss(rho_anchor, count, node_xi, cartesian_weights, measure);
  };

  const int reconstruction_half = reconstruction_width_ / 2;
  for (int pass = 0; pass < 2; ++pass) {
    if (pass == 1) {
      // The collect pass filled the destination vectors with placeholders.
      batch.solve();
      for (std::vector<Real>* destination :
           {&r1_left, &r1_right, &r1_mphi_left, &r1_mphi_right, &r1_bphi_left,
            &r1_bphi_right, &r2_points, &r2_mphi_points, &r2_bphi_points,
            &r3_weights, &r3_mphi_weights, &r3_bphi_weights, &r4_face_to_cell,
            &r5_cell_to_face, &r5_bphi_cell_to_face, &r6_limiter,
            &r6_mphi_limiter, &r6_bphi_limiter}) {
        destination->clear();
      }
    }
  for (int logical = radial_lo_;
       logical < radial_lo_ + radial_count_; ++logical) {
    const Real rho = reduced_cell_radius(grid, logical);

    if (reconstruction_width_ != 0) {
      // R1L and R1R share a physical face but not a reflected coefficient row.
      // Each anchor is obtained from the Grid2D radius accessor of the cell
      // whose face is being reconstructed.
      const Real left_rho = reduced_cell_radius(grid, logical - 1);
      const auto left = solve_row(
          left_rho, reconstruction_width_, -reconstruction_half,
          RadialMomentTarget::point_value, RadialCellMeasure::annular,
          Real{0.5});
      const auto right = solve_row(
          rho, reconstruction_width_, -reconstruction_half,
          RadialMomentTarget::point_value, RadialCellMeasure::annular,
          Real{-0.5});
      const auto mphi_left = solve_row(
          left_rho, reconstruction_width_, -reconstruction_half,
          RadialMomentTarget::point_value,
          RadialCellMeasure::angular_momentum, Real{0.5});
      const auto mphi_right = solve_row(
          rho, reconstruction_width_, -reconstruction_half,
          RadialMomentTarget::point_value,
          RadialCellMeasure::angular_momentum, Real{-0.5});
      const auto bphi_left = solve_row(
          left_rho, reconstruction_width_, -reconstruction_half,
          RadialMomentTarget::point_value, RadialCellMeasure::uniform,
          Real{0.5});
      const auto bphi_right = solve_row(
          rho, reconstruction_width_, -reconstruction_half,
          RadialMomentTarget::point_value, RadialCellMeasure::uniform,
          Real{-0.5});
      // R1 is consumed by every reconstructed component. In particular, the
      // metric-free B_phi flux difference and the radial-momentum positivity
      // fallback do not eliminate the physical r=0 face value, so its moment
      // residual is subject to the same acceptance threshold as every row.
      append_checked_row(r1_left, left, reconstruction_width_, "R1L", logical);
      append_checked_row(r1_right, right, reconstruction_width_, "R1R", logical);
      append_checked_row(
          r1_mphi_left, mphi_left, reconstruction_width_, "R1-mphi-L", logical);
      append_checked_row(
          r1_mphi_right, mphi_right, reconstruction_width_, "R1-mphi-R", logical);
      append_checked_row(
          r1_bphi_left, bphi_left, reconstruction_width_, "R1-bphi-L", logical);
      append_checked_row(
          r1_bphi_right, bphi_right, reconstruction_width_, "R1-bphi-R", logical);

      // R2 has one point-recovery row per radial Gauss node.
      for (int node = 0; node < quadrature_nodes_; ++node) {
        const Real node_xi = scheme_order == 5 ? kMp5TransverseNodes[node]
                                               : kMp7TransverseNodes[node];
        const auto point = solve_row(
            rho, reconstruction_width_, -reconstruction_half,
            RadialMomentTarget::point_value, RadialCellMeasure::annular,
            node_xi);
        const auto mphi_point = solve_row(
            rho, reconstruction_width_, -reconstruction_half,
            RadialMomentTarget::point_value,
            RadialCellMeasure::angular_momentum, node_xi);
        const auto bphi_point = solve_row(
            rho, reconstruction_width_, -reconstruction_half,
            RadialMomentTarget::point_value, RadialCellMeasure::uniform,
            node_xi);
        append_checked_row(r2_points, point, reconstruction_width_, "R2",
                           logical);
        append_checked_row(
            r2_mphi_points, mphi_point, reconstruction_width_, "R2-mphi",
            logical);
        append_checked_row(
            r2_bphi_points, bphi_point, reconstruction_width_, "R2-bphi",
            logical);
      }
      const Real* const nodes = scheme_order == 5 ? kMp5TransverseNodes
                                                  : kMp7TransverseNodes;
      const Real* const weights = scheme_order == 5
          ? kMp5TransverseGaussWeights
          : kMp7TransverseGaussWeights;
      const auto gauss = gauss_row(rho, quadrature_nodes_, nodes, weights,
                                   RadialCellMeasure::annular);
      append_checked_row(r3_weights, gauss, quadrature_nodes_, "R3", logical);
      const auto mphi_gauss = gauss_row(
          rho, quadrature_nodes_, nodes, weights,
          RadialCellMeasure::angular_momentum);
      const auto bphi_gauss = gauss_row(rho, quadrature_nodes_, nodes, weights,
                                        RadialCellMeasure::uniform);
      append_checked_row(
          r3_mphi_weights, mphi_gauss, quadrature_nodes_, "R3-mphi", logical);
      append_checked_row(
          r3_bphi_weights, bphi_gauss, quadrature_nodes_, "R3-bphi", logical);
    }

    // R4 point samples are the centered face sequence around this cell.  Face
    // k is at +1/2 relative to the integer cell index used by the solve API.
    const auto face_to_cell = solve_row(
        rho, collocation_width_, -collocation_width_ / 2,
        RadialMomentTarget::cell_average, RadialCellMeasure::annular,
        Real{0.5});
    append_checked_row(r4_face_to_cell, face_to_cell, collocation_width_, "R4",
                       logical);

    // R5 row `logical` targets the left face of cell `logical`.
    const auto cell_to_face = solve_row(
        rho, collocation_width_, -collocation_width_ / 2,
        RadialMomentTarget::point_value, RadialCellMeasure::annular,
        Real{-0.5});
    // The r=0 collocated scratch value is likewise absent from the annular
    // update and its CT EMF is pinned; only this unused face row's residual may
    // therefore be diagnostic-only.
    append_checked_row(r5_cell_to_face, cell_to_face, collocation_width_, "R5",
                       logical, is_axis_face(grid, logical));
    const auto bphi_cell_to_face = solve_row(
        rho, collocation_width_, -collocation_width_ / 2,
        RadialMomentTarget::point_value, RadialCellMeasure::uniform,
        Real{-0.5});
    append_checked_row(
        r5_bphi_cell_to_face, bphi_cell_to_face, collocation_width_,
        "R5-bphi", logical);

    // R6 first stores the bracketing pair for this cell's right face. The MP
    // v_lc bounds also extrapolate to that face from the two cells on either
    // side, so store their independent slope factors in the same row. Each
    // conserved measure needs its own pair and slope geometry: limiting a
    // native R1 candidate with annular bounds can admit an arbitrary native
    // overshoot even when the ordinary characteristic candidate is bounded.
    const Real right_rho = reduced_cell_radius(grid, logical + 1);
    const auto append_limiter_family = [&](std::vector<Real>& destination,
                                           RadialCellMeasure measure,
                                           const char* family) {
      const auto pair = solve_row(rho, 2, 0,
                                  RadialMomentTarget::point_value, measure,
                                  Real{0.5});
      append_checked_row(destination, pair, 2, family, logical);
      destination.push_back(batch.extrapolation(
          rho, /*neighbor_offset=*/-1, /*face_offset=*/Real{0.5}, measure));
      destination.push_back(batch.extrapolation(
          right_rho, /*neighbor_offset=*/1, /*face_offset=*/Real{-0.5},
          measure));
    };
    append_limiter_family(
        r6_limiter, RadialCellMeasure::annular, "R6-pair");
    append_limiter_family(
        r6_mphi_limiter, RadialCellMeasure::angular_momentum,
        "R6-mphi-pair");
    append_limiter_family(
        r6_bphi_limiter, RadialCellMeasure::uniform, "R6-bphi-pair");
  }
  }

  copy_to_device(r1_left_, r1_left);
  copy_to_device(r1_right_, r1_right);
  copy_to_device(r1_mphi_left_, r1_mphi_left);
  copy_to_device(r1_mphi_right_, r1_mphi_right);
  copy_to_device(r1_bphi_left_, r1_bphi_left);
  copy_to_device(r1_bphi_right_, r1_bphi_right);
  copy_to_device(r2_points_, r2_points);
  copy_to_device(r2_mphi_points_, r2_mphi_points);
  copy_to_device(r2_bphi_points_, r2_bphi_points);
  copy_to_device(r3_weights_, r3_weights);
  copy_to_device(r3_mphi_weights_, r3_mphi_weights);
  copy_to_device(r3_bphi_weights_, r3_bphi_weights);
  copy_to_device(r4_face_to_cell_, r4_face_to_cell);
  copy_to_device(r5_cell_to_face_, r5_cell_to_face);
  copy_to_device(r5_bphi_cell_to_face_, r5_bphi_cell_to_face);
  copy_to_device(r6_limiter_, r6_limiter);
  copy_to_device(r6_mphi_limiter_, r6_mphi_limiter);
  copy_to_device(r6_bphi_limiter_, r6_bphi_limiter);
}

RadialTablesView RadialTables::view() const noexcept {
  return RadialTablesView{
      active_,
      scheme_order_,
      radial_lo_,
      radial_count_,
      reconstruction_width_,
      quadrature_nodes_,
      collocation_width_,
      r1_left_.device_ptr(),
      r1_right_.device_ptr(),
      r1_mphi_left_.device_ptr(),
      r1_mphi_right_.device_ptr(),
      r1_bphi_left_.device_ptr(),
      r1_bphi_right_.device_ptr(),
      r2_points_.device_ptr(),
      r2_mphi_points_.device_ptr(),
      r2_bphi_points_.device_ptr(),
      r3_weights_.device_ptr(),
      r3_mphi_weights_.device_ptr(),
      r3_bphi_weights_.device_ptr(),
      r4_face_to_cell_.device_ptr(),
      r5_cell_to_face_.device_ptr(),
      r5_bphi_cell_to_face_.device_ptr(),
      r6_limiter_.device_ptr(),
      r6_mphi_limiter_.device_ptr(),
      r6_bphi_limiter_.device_ptr()};
}

RadialTablesView RadialTables::host_view() const noexcept {
  const auto host_data = [](const std::vector<Real>& rows) noexcept {
    return rows.empty() ? nullptr : rows.data();
  };
  return RadialTablesView{
      active_,
      scheme_order_,
      radial_lo_,
      radial_count_,
      reconstruction_width_,
      quadrature_nodes_,
      collocation_width_,
      host_data(r1_left_host_),
      host_data(r1_right_host_),
      host_data(r1_mphi_left_host_),
      host_data(r1_mphi_right_host_),
      host_data(r1_bphi_left_host_),
      host_data(r1_bphi_right_host_),
      host_data(r2_points_host_),
      host_data(r2_mphi_points_host_),
      host_data(r2_bphi_points_host_),
      host_data(r3_weights_host_),
      host_data(r3_mphi_weights_host_),
      host_data(r3_bphi_weights_host_),
      host_data(r4_face_to_cell_host_),
      host_data(r5_cell_to_face_host_),
      host_data(r5_bphi_cell_to_face_host_),
      host_data(r6_limiter_host_),
      host_data(r6_mphi_limiter_host_),
      host_data(r6_bphi_limiter_host_)};
}

}  // namespace quasar::numerics
