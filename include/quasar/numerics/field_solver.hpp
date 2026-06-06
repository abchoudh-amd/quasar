#pragma once

#include "quasar/core/registry.hpp"
#include "quasar/core/yee_field.hpp"

namespace quasar::numerics {

// Generalized field-solver interface. A field solver advances a field state in
// two half-updates: advance_b (source-free) and advance_e (driven by a source
// term, e.g. the deposited current for EM-PIC). It is templated over the field
// state type `Field` and the source type `Source` so a future non-PIC consumer
// (e.g. an MHD module with its own state/flux types) can reuse the same contract
// without touching the EM-PIC instantiation below.
template <class Field, class Source>
class IFieldSolverT {
 public:
  virtual ~IFieldSolverT() = default;
  virtual void advance_b(Field& f, Real dt) const = 0;
  virtual void advance_e(Field& f, const Source& src, Real dt) const = 0;
};

// Behavior-preserving alias for the EM-PIC instantiation. Existing clients,
// concrete solvers (YeeFdtd2D / YeeFdtdCyl2D), and the registry registrations all
// keep using `IFieldSolver` unchanged — it resolves to the same concrete type as
// before, so the registry key (Registry<IFieldSolver>) and the deck-facing names
// ("yee_o2" / "yee_o4" / "yee_cyl_o2") are byte-for-byte equivalent in behavior.
//
// EM-PIC remains phrased in the concrete YeeField2D/JField2D types, and the
// concrete solvers' definitions + registrations still live in
// src/physics/pic/pic_solver.cpp; see the axis-orthogonality note in CLAUDE.md.
using IFieldSolver = IFieldSolverT<YeeField2D<Real>, JField2D<Real>>;

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
