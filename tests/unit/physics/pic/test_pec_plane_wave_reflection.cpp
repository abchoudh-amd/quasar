// PEC plane-wave reflection.
//
// A real reflection test requires the field-ghost path (QUASAR_PIC_FIELD_GHOSTS)
// so the PEC wall actually imposes its boundary on the FDTD stencil. That path
// is wired but disabled by default: the stencil still reads via periodic_index,
// so PEC field walls are inert and a reflection assertion cannot pass yet (see
// plans/field_bc_heisenbug.md for the remaining ghost-aware-stencil work). Until
// then this verifies the PEC field BC constructs and its ghost fill is callable;
// it is skipped when field ghosts are off so it is not a silent false pass.

#include "quasar/backend/device.hpp"
#include "quasar/boundary/wall.hpp"
#include "quasar/core/yee_field.hpp"

#include <gtest/gtest.h>

TEST(PicPecPlaneWaveReflection, FieldBcConstructsAndGhostFillRuns) {
#ifndef QUASAR_PIC_FIELD_GHOSTS
  GTEST_SKIP() << "PEC field walls are inert without QUASAR_PIC_FIELD_GHOSTS; "
                  "reflection test pending the ghost-aware stencil "
                  "(plans/field_bc_heisenbug.md).";
#else
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  quasar::YeeField2D<double> field{quasar::Grid2D{8, 8, 1.0, 1.0, 0.0, 0.0, 2}};
  quasar::boundary::PecFieldBC bc;
  EXPECT_NO_THROW(bc.fill_ghosts(field, quasar::Side::x_lo));
#endif
}
