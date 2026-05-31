#pragma once

#include "quasar/core/registry.hpp"
#include "quasar/core/yee_field.hpp"

namespace quasar::numerics {

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
