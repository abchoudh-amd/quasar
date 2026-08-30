// Micro-benchmark sweep for the Biot-Savart HIP kernels.
//
// Each fixture builds a closed circular loop with N straight segments and a
// regular grid of M observation points, warms the HIP kernel once, and then
// measures the per-iteration cost of evaluate_B (and evaluate_grad_B).
//
// Usage:
//
//   cmake --preset hip-gfx942-release -DQUASAR_BUILD_BENCHMARKS=ON
//   cmake --build build/hip-gfx942-release --target quasar_bench_biot_savart
//   build/hip-gfx942-release/benchmarks/micro/quasar_bench_biot_savart \
//       --benchmark_format=console
//
// `items_processed` reports N * M = the number of (segment, observation)
// pairs touched per iteration, which makes the per-pair throughput
// directly comparable across (N, M) configurations.

#include "quasar/core/device_observations.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/geometry.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>

namespace {

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::magnetostatics::BiotSavartEvaluator;
using ::quasar::magnetostatics::circular_loop;
using ::quasar::magnetostatics::ConductorSystem;
using ::quasar::magnetostatics::ObservationGrid;
using ::quasar::magnetostatics::PointCloud;

struct Fixture {
  ConductorSystem                cs;
  PointCloud                     pc;
  ::quasar::core::DevicePointCloud device_pc;
};

Fixture build(int N, int M) {
  Fixture f;
  f.cs.add(circular_loop(/*center=*/Vec3{0, 0, 0},
                         /*axis=*/Vec3{0, 0, 1},
                         /*radius_m=*/Real{0.1},
                         /*n_segments=*/N,
                         /*current_A=*/Real{1.0},
                         /*name=*/"loop"));

  // Cubic observation grid sized so |grid| ~ M.
  int side = 1;
  while ((side + 1) * (side + 1) * (side + 1) <= M) ++side;
  if (side < 1) side = 1;

  ObservationGrid g;
  g.origin  = Vec3{Real{-0.2}, Real{-0.2}, Real{-0.2}};
  g.spacing = Vec3{Real{0.4} / static_cast<Real>(side - 1 == 0 ? 1 : side - 1),
                   Real{0.4} / static_cast<Real>(side - 1 == 0 ? 1 : side - 1),
                   Real{0.4} / static_cast<Real>(side - 1 == 0 ? 1 : side - 1)};
  g.dims = {side, side, side};
  f.pc   = g.to_point_cloud();
  // The evaluator now consumes device-resident points, so the upload happens
  // once here rather than inside every timed iteration. That is the honest
  // measurement: what the loop below times is the kernel, which is what the
  // fixture varies N and M to stress.
  f.device_pc = ::quasar::core::DevicePointCloud::upload(f.pc);
  return f;
}

}  // namespace

static void BM_BiotSavart_B(benchmark::State& state) {
  const int N = static_cast<int>(state.range(0));
  const int M = static_cast<int>(state.range(1));
  const Fixture fx = build(N, M);
  const BiotSavartEvaluator eval;

  // Warmup (kernel JIT, hipMalloc paths).
  benchmark::DoNotOptimize(eval.evaluate_B(fx.cs, fx.device_pc));

  for (auto _ : state) {
    auto B = eval.evaluate_B(fx.cs, fx.device_pc);
    benchmark::DoNotOptimize(B);
  }

  const std::size_t M_actual = fx.pc.size();
  state.SetItemsProcessed(state.iterations()
                          * static_cast<std::int64_t>(N)
                          * static_cast<std::int64_t>(M_actual));
  state.counters["N"]        = static_cast<double>(N);
  state.counters["M_actual"] = static_cast<double>(M_actual);
}
BENCHMARK(BM_BiotSavart_B)
    ->ArgsProduct({{64, 256, 1024, 4096},
                    {512, 8192, 65536}})
    ->Unit(benchmark::kMillisecond);

static void BM_BiotSavart_gradB(benchmark::State& state) {
  const int N = static_cast<int>(state.range(0));
  const int M = static_cast<int>(state.range(1));
  const Fixture fx = build(N, M);
  const BiotSavartEvaluator eval;

  benchmark::DoNotOptimize(eval.evaluate_grad_B(fx.cs, fx.device_pc));

  for (auto _ : state) {
    auto G = eval.evaluate_grad_B(fx.cs, fx.device_pc);
    benchmark::DoNotOptimize(G);
  }

  const std::size_t M_actual = fx.pc.size();
  state.SetItemsProcessed(state.iterations()
                          * static_cast<std::int64_t>(N)
                          * static_cast<std::int64_t>(M_actual));
  state.counters["N"]        = static_cast<double>(N);
  state.counters["M_actual"] = static_cast<double>(M_actual);
}
BENCHMARK(BM_BiotSavart_gradB)
    ->ArgsProduct({{64, 256, 1024},
                    {512, 8192}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
