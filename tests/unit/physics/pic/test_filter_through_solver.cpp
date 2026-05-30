// The current-smoothing filter pipeline must be reachable from EmPicConfig: a
// solver configured with a binomial filter has to actually smooth the deposited
// current each step, not silently no-op (the pipeline was previously unreachable
// because nothing populated EmPic2D3V::filters_).

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace {

// Steps a single drifting particle once and returns the peak |Jz| on the grid,
// with `n_passes` binomial filter passes configured (0 = no filter).
double peak_jz_after_step(int n_passes) {
  quasar::Grid2D g{32, 32, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, 1};
  if (n_passes > 0) {
    cfg.filters.push_back(quasar::pic::FilterSpec{"binomial", n_passes});
  }
  quasar::pic::EmPic2D3V solver{cfg};

  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{"q", 1.0, 1.0, 1}};
  // vz drives Jz, which the binomial filter smooths in-plane.
  sp.set_host_particles({0.51}, {0.52}, {0.1}, {0.0}, {1.0}, {1.0});
  solver.add_species(std::move(sp));

  solver.step(0.05);

  auto& J = solver.current();
  std::vector<double> jz(g.storage_size());
  J.jz.copy_to_host(jz.data(), jz.size());
  double peak = 0.0;
  for (double v : jz) peak = std::max(peak, std::abs(v));
  return peak;
}

}  // namespace

TEST(PicFilterThroughSolver, BinomialFilterReducesCurrentPeak) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const double peak_unfiltered = peak_jz_after_step(0);
  const double peak_filtered = peak_jz_after_step(2);

  EXPECT_GT(peak_unfiltered, 0.0) << "no current deposited";
  // Smoothing spreads the spike, so the peak must drop appreciably.
  EXPECT_LT(peak_filtered, 0.9 * peak_unfiltered)
      << "binomial filter did not smooth the current (pipeline not wired): "
      << peak_filtered << " vs " << peak_unfiltered;
}
