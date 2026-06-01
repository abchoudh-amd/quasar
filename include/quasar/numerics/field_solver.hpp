#pragma once

#include "quasar/core/registry.hpp"
#include "quasar/core/yee_field.hpp"

namespace quasar::numerics {

// EM-PIC-specific by design: this interface (and YeeFdtd2D) is phrased in the
// concrete YeeField2D/JField2D types and its definitions + registrations live in
// src/physics/pic/pic_solver.cpp. With EM-PIC the only consumer, it is not yet
// templated over field types; see the axis-orthogonality note in CLAUDE.md.
class IFieldSolver {
 public:
  virtual ~IFieldSolver() = default;
  virtual void advance_b(YeeField2D<Real>& fields, Real dt) const = 0;
  virtual void advance_e(YeeField2D<Real>& fields, const JField2D<Real>& current,
                         Real dt) const = 0;
};

template <int Order>
class YeeFdtd2D final : public IFieldSolver {
 public:
  void advance_b(YeeField2D<Real>& fields, Real dt) const override;
  void advance_e(YeeField2D<Real>& fields, const JField2D<Real>& current,
                 Real dt) const override;
};

}  // namespace quasar::numerics

// Registers a concrete field solver under a deck-facing name (e.g. "yee_o2").
#define QUASAR_REGISTER_FIELD_SOLVER(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::numerics::IFieldSolver, Name, Class)
