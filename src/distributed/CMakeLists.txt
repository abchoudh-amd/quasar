quasar_add_module(distributed DETACHED SOURCES
  topology.cpp
  device_mapping.cpp
  mpi_device_mapping.cpp
  collective_error.cpp
  mpi_runtime.cpp
  runtime_session.cpp
  worker_pool.cpp
  transport.cpp
  mhd_halo.cpp
  mhd_runtime.cpp
  pic_runtime.cpp
  checkpoint.cpp
  diagnostics.cpp
)
add_library(quasar::distributed ALIAS quasar_distributed)
target_compile_definitions(quasar_distributed
  PUBLIC
    QUASAR_DISTRIBUTED_AVAILABLE=1
)
if(QUASAR_DISTRIBUTED_TEST_HOOKS)
  target_compile_definitions(quasar_distributed
    PUBLIC
      QUASAR_DISTRIBUTED_TEST_HOOKS=1
  )
endif()

# These dependencies are intentionally attached only to the opt-in
# distributed target.  In particular, quasar::core does not acquire MPI or
# HDF5 usage requirements when distributed support is enabled.
target_link_libraries(quasar_distributed
  PRIVATE
    MPI::MPI_CXX
    Threads::Threads
    HDF5::HDF5
    quasar::core
)
