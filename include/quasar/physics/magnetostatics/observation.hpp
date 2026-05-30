#pragma once

// The observation-point sets are axis-neutral and now live in core. This header
// re-exports them into the magnetostatics namespace so existing magnetostatics
// call sites (and the registered evaluators) keep compiling unchanged; the
// numerics field-evaluator interface depends on the core types directly.

#include "quasar/core/observations.hpp"

namespace quasar::magnetostatics {

using core::LineProbe;
using core::ObservationGrid;
using core::PlaneSlice;
using core::PointCloud;
using core::PointSoA;

}  // namespace quasar::magnetostatics
