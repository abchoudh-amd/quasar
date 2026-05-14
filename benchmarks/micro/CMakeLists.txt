if(NOT QUASAR_BUILD_BENCHMARKS OR NOT QUASAR_ENABLE_HIP)
  return()
endif()

add_executable(quasar_bench_biot_savart biot_savart_bench.cpp)

# Pulls quasar/backend/* (HIP runtime headers) transitively; compile through
# the HIP compiler so the runtime library links in.
set_source_files_properties(biot_savart_bench.cpp PROPERTIES LANGUAGE HIP)

target_link_libraries(quasar_bench_biot_savart
  PRIVATE
    quasar::core
    benchmark::benchmark
)

# Sanity smoke test that the benchmark binary actually runs; the real perf
# sweep is intended to be run by hand with `--benchmark_format=console`
# (or `--benchmark_format=json` for archival).
if(QUASAR_BUILD_TESTING)
  add_test(
    NAME    benchmarks_biot_savart_smoke
    COMMAND quasar_bench_biot_savart
            --benchmark_min_time=0.005s
            --benchmark_filter=BM_BiotSavart_B/64/1024
  )
endif()
