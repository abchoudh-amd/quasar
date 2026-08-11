#pragma once

// Solver-owned radial finite-volume coefficient tables.
//
// Cylindrical components are finite-volume averages under their evolution
// measures, so every high-order operation whose stencil runs in the radial
// direction needs coefficients tied to both the global radius and the stored
// component.  RadialTables builds and retains those rows on the host and
// mirrors them in device storage.  RadialTablesView is the small, trivially
// copyable kernel ABI: an inactive/default view denotes Cartesian geometry.

#include "quasar/backend/memory.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"

#include <cstddef>
#include <type_traits>
#include <vector>

namespace quasar::numerics {

struct RadialTablesView {
  int active{0};
  int scheme_order{0};
  int radial_lo{0};
  int radial_count{0};
  int reconstruction_width{0};
  int quadrature_nodes{0};
  int collocation_width{0};

  // R1: cell averages -> a face point, with genuinely distinct rows for the
  // state reconstructed from the left and right of the face. Most components
  // use annular |r| dr averages; m_phi and B_phi have their own conserved
  // measures (r^2 dr and dr, respectively).
  const Real* r1_left{nullptr};
  const Real* r1_right{nullptr};
  const Real* r1_mphi_left{nullptr};
  const Real* r1_mphi_right{nullptr};
  const Real* r1_bphi_left{nullptr};
  const Real* r1_bphi_right{nullptr};
  // R2: equation-native cell averages -> each Gauss point of the central
  // radial cell.
  const Real* r2_points{nullptr};
  const Real* r2_mphi_points{nullptr};
  const Real* r2_bphi_points{nullptr};
  // R3: Gauss weights with the radial measure folded in and renormalized.
  const Real* r3_weights{nullptr};
  const Real* r3_mphi_weights{nullptr};
  const Real* r3_bphi_weights{nullptr};
  // R4: radial face-point samples -> a ring-cell average.
  const Real* r4_face_to_cell{nullptr};
  // R5: cell averages -> a centered radial face point.  Staggered in-plane
  // fields use annular rows; cell-centred B_phi (evolved or prescribed) uses
  // the uniform-measure family.
  const Real* r5_cell_to_face{nullptr};
  const Real* r5_bphi_cell_to_face{nullptr};
  // R6 row i (stride 4): [0:2] is the two-cell, linear-exact pair for the face
  // between cells i and i+1. Entries 2 and 3 are the one-sided v_lc slope
  // factors for the left and right states at that face, respectively. As for
  // R1, the ordinary, m_phi, and B_phi families use their conserved measures.
  const Real* r6_limiter{nullptr};
  const Real* r6_mphi_limiter{nullptr};
  const Real* r6_bphi_limiter{nullptr};

  QUASAR_HOST_DEVICE constexpr bool contains(int logical_index) const noexcept {
    return logical_index >= radial_lo &&
           logical_index < radial_lo + radial_count;
  }

  QUASAR_HOST_DEVICE constexpr int row_index(
      int logical_index) const noexcept {
    return logical_index - radial_lo;
  }

  QUASAR_HOST_DEVICE constexpr const Real* r1_left_row(
      int face) const noexcept {
    return r1_left + static_cast<std::size_t>(row_index(face)) *
                         static_cast<std::size_t>(reconstruction_width);
  }

  QUASAR_HOST_DEVICE constexpr const Real* r1_right_row(
      int face) const noexcept {
    return r1_right + static_cast<std::size_t>(row_index(face)) *
                          static_cast<std::size_t>(reconstruction_width);
  }

  QUASAR_HOST_DEVICE constexpr const Real* r1_mphi_left_row(
      int face) const noexcept {
    return r1_mphi_left + static_cast<std::size_t>(row_index(face)) *
                               static_cast<std::size_t>(reconstruction_width);
  }

  QUASAR_HOST_DEVICE constexpr const Real* r1_mphi_right_row(
      int face) const noexcept {
    return r1_mphi_right + static_cast<std::size_t>(row_index(face)) *
                                static_cast<std::size_t>(reconstruction_width);
  }

  QUASAR_HOST_DEVICE constexpr const Real* r1_bphi_left_row(
      int face) const noexcept {
    return r1_bphi_left + static_cast<std::size_t>(row_index(face)) *
                               static_cast<std::size_t>(reconstruction_width);
  }

  QUASAR_HOST_DEVICE constexpr const Real* r1_bphi_right_row(
      int face) const noexcept {
    return r1_bphi_right + static_cast<std::size_t>(row_index(face)) *
                                static_cast<std::size_t>(reconstruction_width);
  }

  QUASAR_HOST_DEVICE constexpr const Real* r2_row(
      int cell, int node) const noexcept {
    const std::size_t row =
        static_cast<std::size_t>(row_index(cell)) *
            static_cast<std::size_t>(quadrature_nodes) +
        static_cast<std::size_t>(node);
    return r2_points + row * static_cast<std::size_t>(reconstruction_width);
  }

  QUASAR_HOST_DEVICE constexpr const Real* r2_mphi_row(
      int cell, int node) const noexcept {
    const std::size_t row =
        static_cast<std::size_t>(row_index(cell)) *
            static_cast<std::size_t>(quadrature_nodes) +
        static_cast<std::size_t>(node);
    return r2_mphi_points +
           row * static_cast<std::size_t>(reconstruction_width);
  }

  QUASAR_HOST_DEVICE constexpr const Real* r2_bphi_row(
      int cell, int node) const noexcept {
    const std::size_t row =
        static_cast<std::size_t>(row_index(cell)) *
            static_cast<std::size_t>(quadrature_nodes) +
        static_cast<std::size_t>(node);
    return r2_bphi_points +
           row * static_cast<std::size_t>(reconstruction_width);
  }

  QUASAR_HOST_DEVICE constexpr const Real* r3_row(int cell) const noexcept {
    return r3_weights + static_cast<std::size_t>(row_index(cell)) *
                            static_cast<std::size_t>(quadrature_nodes);
  }

  QUASAR_HOST_DEVICE constexpr const Real* r3_mphi_row(
      int cell) const noexcept {
    return r3_mphi_weights +
           static_cast<std::size_t>(row_index(cell)) *
               static_cast<std::size_t>(quadrature_nodes);
  }

  QUASAR_HOST_DEVICE constexpr const Real* r3_bphi_row(
      int cell) const noexcept {
    return r3_bphi_weights +
           static_cast<std::size_t>(row_index(cell)) *
               static_cast<std::size_t>(quadrature_nodes);
  }

  QUASAR_HOST_DEVICE constexpr const Real* r4_row(int cell) const noexcept {
    return r4_face_to_cell +
           static_cast<std::size_t>(row_index(cell)) *
               static_cast<std::size_t>(collocation_width);
  }

  QUASAR_HOST_DEVICE constexpr const Real* r5_row(int face) const noexcept {
    return r5_cell_to_face +
           static_cast<std::size_t>(row_index(face)) *
               static_cast<std::size_t>(collocation_width);
  }

  QUASAR_HOST_DEVICE constexpr const Real* r5_bphi_row(
      int face) const noexcept {
    return r5_bphi_cell_to_face +
           static_cast<std::size_t>(row_index(face)) *
               static_cast<std::size_t>(collocation_width);
  }

  QUASAR_HOST_DEVICE constexpr const Real* r6_row(int cell) const noexcept {
    return r6_limiter +
           static_cast<std::size_t>(row_index(cell)) * std::size_t{4};
  }

  QUASAR_HOST_DEVICE constexpr const Real* r6_mphi_row(
      int cell) const noexcept {
    return r6_mphi_limiter +
           static_cast<std::size_t>(row_index(cell)) * std::size_t{4};
  }

  QUASAR_HOST_DEVICE constexpr const Real* r6_bphi_row(
      int cell) const noexcept {
    return r6_bphi_limiter +
           static_cast<std::size_t>(row_index(cell)) * std::size_t{4};
  }
};

static_assert(std::is_trivially_copyable_v<RadialTablesView>,
              "RadialTablesView must remain a by-value kernel argument");

class RadialTables {
 public:
  RadialTables() noexcept = default;

  // Construct active cylindrical tables for the resolved working grid.  The
  // grid must already contain the reconstruction-selected nghost.  Supported
  // scheme orders are 2, 5, and 7; order 2 owns only the collocation/limiter
  // families because it has no MP reconstruction or transverse Gauss rule.
  RadialTables(const Grid2D& grid, int scheme_order);

  RadialTables(const RadialTables&) = delete;
  RadialTables& operator=(const RadialTables&) = delete;
  RadialTables(RadialTables&&) noexcept = default;
  RadialTables& operator=(RadialTables&&) noexcept = default;

  // Device-facing view for kernel launches.
  RadialTablesView view() const noexcept;
  // Host-side algorithms must use this view rather than dereference the
  // device pointers returned by view(). The coefficient rows are identical
  // and remain valid for the lifetime of this owner.
  RadialTablesView host_view() const noexcept;
  bool active() const noexcept { return active_ != 0; }

 private:
  int active_{0};
  int scheme_order_{0};
  int radial_lo_{0};
  int radial_count_{0};
  int reconstruction_width_{0};
  int quadrature_nodes_{0};
  int collocation_width_{0};

  std::vector<Real> r1_left_host_{};
  std::vector<Real> r1_right_host_{};
  std::vector<Real> r1_mphi_left_host_{};
  std::vector<Real> r1_mphi_right_host_{};
  std::vector<Real> r1_bphi_left_host_{};
  std::vector<Real> r1_bphi_right_host_{};
  std::vector<Real> r2_points_host_{};
  std::vector<Real> r2_mphi_points_host_{};
  std::vector<Real> r2_bphi_points_host_{};
  std::vector<Real> r3_weights_host_{};
  std::vector<Real> r3_mphi_weights_host_{};
  std::vector<Real> r3_bphi_weights_host_{};
  std::vector<Real> r4_face_to_cell_host_{};
  std::vector<Real> r5_cell_to_face_host_{};
  std::vector<Real> r5_bphi_cell_to_face_host_{};
  std::vector<Real> r6_limiter_host_{};
  std::vector<Real> r6_mphi_limiter_host_{};
  std::vector<Real> r6_bphi_limiter_host_{};

  backend::DeviceBuffer<Real> r1_left_{};
  backend::DeviceBuffer<Real> r1_right_{};
  backend::DeviceBuffer<Real> r1_mphi_left_{};
  backend::DeviceBuffer<Real> r1_mphi_right_{};
  backend::DeviceBuffer<Real> r1_bphi_left_{};
  backend::DeviceBuffer<Real> r1_bphi_right_{};
  backend::DeviceBuffer<Real> r2_points_{};
  backend::DeviceBuffer<Real> r2_mphi_points_{};
  backend::DeviceBuffer<Real> r2_bphi_points_{};
  backend::DeviceBuffer<Real> r3_weights_{};
  backend::DeviceBuffer<Real> r3_mphi_weights_{};
  backend::DeviceBuffer<Real> r3_bphi_weights_{};
  backend::DeviceBuffer<Real> r4_face_to_cell_{};
  backend::DeviceBuffer<Real> r5_cell_to_face_{};
  backend::DeviceBuffer<Real> r5_bphi_cell_to_face_{};
  backend::DeviceBuffer<Real> r6_limiter_{};
  backend::DeviceBuffer<Real> r6_mphi_limiter_{};
  backend::DeviceBuffer<Real> r6_bphi_limiter_{};
};

}  // namespace quasar::numerics
