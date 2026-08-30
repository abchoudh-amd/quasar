#pragma once

#include "quasar/numerics/field_evaluator.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace quasar::analytic_fields {

class FileGridEvaluator final : public numerics::IFieldEvaluator {
 public:
  FileGridEvaluator() = default;
  explicit FileGridEvaluator(std::string path);

  // Configure a rectilinear nodal field through the registry seam. Required
  // keys are:
  //   origin  = [x0,y0,z0]
  //   spacing = [dx,dy,dz]
  //   dims    = [nx,ny,nz] (positive exact integers)
  //   values  = flattened [Bx,By,Bz,...], x-fastest
  // This is also the bridge used by Python to load NumPy .npz maps without
  // embedding a NumPy/ZIP parser in the C++ core.
  void configure(const numerics::EvaluatorParams& params) override;

  core::DeviceVectorField evaluate_B(
      const core::IFieldSource&,
      const core::DevicePointCloud& observations) const override;
  // A complete Jacobian is determined only when the configured map has at
  // least two nodes on every axis.  A singleton axis is one geometric plane,
  // not an assertion that the field is invariant normal to that plane.
  bool provides_grad_B() const noexcept override;
  core::DeviceTensorField evaluate_grad_B(
      const core::IFieldSource&,
      const core::DevicePointCloud& observations) const override;

  const std::string& path() const noexcept { return path_; }
  bool configured() const noexcept { return configured_; }

 private:
  // Host staging only: the parsed deck or file contents on the way to the
  // device. `values` is released once set_grid() has uploaded it, so the
  // configured evaluator holds the map exactly once, on the device.
  struct GridData {
    Vec3 origin{};
    Vec3 spacing{Real{1}, Real{1}, Real{1}};
    std::array<std::size_t, 3> dims{0, 0, 0};
    std::vector<Vec3> values{};
    Real divergence_tolerance{Real{1e-6}};
  };

  static GridData load_text_grid(const std::string& path);
  // Uploads the map, then runs the finiteness and solenoidality sweeps on the
  // device. Cheap host checks that do not scale with the node count (finite
  // origin/spacing, matching value count, non-collapsing coordinates) stay on
  // the host, since they are scalar configuration math.
  void set_grid(GridData grid);

  std::string path_{};
  GridData grid_{};
  // Device-resident node values, SoA. Length dims[0]*dims[1]*dims[2] each.
  backend::DeviceBuffer<Real> vx_{}, vy_{}, vz_{};
  bool configured_{false};
};

}  // namespace quasar::analytic_fields
