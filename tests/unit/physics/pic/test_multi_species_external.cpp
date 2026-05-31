// Smoke test: two species (proton + electron-mass surrogate for muon) drift
// in an external B-field sampled from a single Biot–Savart loop. Confirms the
// end-to-end wiring used by the Python `quasar.pic.cli` square_toroid_pic
// example: add_species, sample_external_field, step() with multiple species
// in a non-uniform external field.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/geometry.hpp"
#include "quasar/physics/pic/diagnostics.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace {

constexpr quasar::Real kQE     = 1.602176634e-19;
constexpr quasar::Real kMProt  = 1.67262192369e-27;
constexpr quasar::Real kMMuon  = 1.883531627e-28;

quasar::pic::ParticleSpecies make_species(const char* name, quasar::Real q,
                                          quasar::Real m, std::size_t n,
                                          quasar::Real x0, quasar::Real y0,
                                          quasar::Real vy) {
  quasar::pic::ParticleSpecies sp{quasar::pic::SpeciesConfig{name, q, m, n}};
  std::vector<quasar::Real> x(n, x0), y(n, y0);
  std::vector<quasar::Real> vx(n, 0), vyv(n, vy), vz(n, 0);
  std::vector<quasar::Real> w(n, 1.0);
  sp.set_host_particles(x, y, vx, vyv, vz, w);
  return sp;
}

}  // namespace

TEST(PicMultiSpeciesExternal, ProtonAndMuonStepInBiotSavartField) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // Small grid centered on a circular loop placed at the origin. The loop has
  // axis ẑ so it produces a strong Bz on the equatorial slice the PIC sees.
  quasar::Grid2D g{16, 16, 0.10, 0.10, -0.05, -0.05, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};

  quasar::magnetostatics::ConductorSystem cs;
  cs.add(quasar::magnetostatics::circular_loop(
      quasar::Vec3{0, 0, 0}, quasar::Vec3{0, 0, 1},
      /*radius_m=*/0.04, /*n_segments=*/64, /*current_A=*/1000.0));
  quasar::magnetostatics::BiotSavartEvaluator eval;
  quasar::pic::sample_external_field(eval, cs, solver.external_fields());

  solver.add_species(make_species("H+",  +kQE, kMProt, 8, 0.0, 0.0, 1.0e4));
  solver.add_species(make_species("mu-", -kQE, kMMuon, 8, 0.0, 0.0, 1.0e4));
  ASSERT_EQ(solver.species().size(), 2u);

  const quasar::Real dt = 1.0e-12;
  for (int s = 0; s < 8; ++s) solver.step(dt);

  // Both species should still be alive.
  EXPECT_EQ(quasar::pic::alive_count(solver.species()[0]), 8u);
  EXPECT_EQ(quasar::pic::alive_count(solver.species()[1]), 8u);

  // Particles should have responded to Bz (cyclotron rotation gives them
  // a non-zero vx even though they started with vx=0).
  auto sH = solver.species()[0].to_host();
  auto sM = solver.species()[1].to_host();
  ASSERT_EQ(sH.vx.size(), 8u);
  ASSERT_EQ(sM.vx.size(), 8u);
  bool h_moved = false, m_moved = false;
  for (auto v : sH.vx) if (std::abs(v) > 0) { h_moved = true; break; }
  for (auto v : sM.vx) if (std::abs(v) > 0) { m_moved = true; break; }
  EXPECT_TRUE(h_moved);
  EXPECT_TRUE(m_moved);

  // Sanity check that the external Bz at the bore is non-trivial.
  std::vector<quasar::Real> bz(solver.external_fields().bz.size());
  solver.external_fields().bz.copy_to_host(bz.data(), bz.size());
  quasar::Real max_abs = 0;
  for (auto v : bz) max_abs = std::max(max_abs, std::abs(v));
  EXPECT_GT(max_abs, 1.0e-4);
}
