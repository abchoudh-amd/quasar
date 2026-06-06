// Anchor translation unit for the generalized field-solver interface.
//
// The IFieldSolverT<Field, Source> template and its EM-PIC alias
// (IFieldSolver = IFieldSolverT<YeeField2D<Real>, JField2D<Real>>) are fully
// header-defined in include/quasar/numerics/field_solver.hpp. The concrete
// EM-PIC solvers (YeeFdtd2D / YeeFdtdCyl2D) and their registry registrations
// continue to live in src/physics/pic/pic_solver.cpp — see the axis-orthogonality
// note in CLAUDE.md.
//
// This TU exists so the numerics module enumerates field_solver.cpp; it gives a
// future non-PIC consumer (e.g. an MHD module) a home in src/numerics/ for any
// out-of-line bits of the generalized base without disturbing the EM-PIC path.
#include "quasar/numerics/field_solver.hpp"
