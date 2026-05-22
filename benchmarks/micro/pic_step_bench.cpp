#include "quasar/physics/pic/pic_solver.hpp"

#include <benchmark/benchmark.h>

static void BM_PicStepConstruct(benchmark::State& state) {
  for (auto _ : state) {
    quasar::pic::EmPicConfig cfg;
    cfg.grid = quasar::Grid2D{static_cast<int>(state.range(0)),
                              static_cast<int>(state.range(0)), 1.0, 1.0};
    quasar::pic::EmPic2D3V solver{cfg};
    benchmark::DoNotOptimize(solver.grid().nx);
  }
}

BENCHMARK(BM_PicStepConstruct)->Arg(16)->Arg(64);

BENCHMARK_MAIN();
